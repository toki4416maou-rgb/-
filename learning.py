"""
Teacher-free structural search and Autonomous X Discovery for Akane v1.0.

No teacher_plan exists.

Training:
    observation
    -> candidate primitive ranking
    -> bounded plan search
    -> execute candidate
    -> Verifier checks only final answer
    -> successful execution trace becomes evidence
    -> routing weights update
    -> recurring trace is promoted into a persistent recipe X

Inference:
    observation
    -> retrieve known recipe X OR synthesize sparse primitive plan
    -> choose highest learned score
    -> execute once

The expected answer is used only by the training Verifier, never to reveal the
correct primitive sequence.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import random

from core import (
    CODE_TO_PRIMITIVE,
    REGISTRY,
    ObservationSignature,
    PrimitiveX,
    observe,
    pack_recipe,
    recipe_bytes,
    run_lama_x,
    unpack_recipe,
)
from memory import XStore


ROLE_ORDER = ("filter", "compare", "extreme", "terminal")


@dataclass
class EpisodeResult:
    correct: bool
    value: Any
    expected: Any
    lama_x: int
    primitive_codes: Tuple[int, ...]
    reward: float
    tried_plans: int
    discovered: bool
    promoted: bool
    used_known_recipe: bool
    active_x: int
    expanded_primitives: int
    steps: int


class LamaSystem:
    def __init__(
        self,
        store: Optional[XStore] = None,
        seed: int = 42,
        learning_rate: float = 0.22,
        promotion_threshold: int = 6,
        search_budget: int = 32,
        per_role_candidates: int = 8,
    ):
        self.store = store or XStore()
        self.rng = random.Random(seed)
        self.learning_rate = float(learning_rate)
        self.promotion_threshold = int(promotion_threshold)
        self.search_budget = int(search_budget)
        self.per_role_candidates = int(per_role_candidates)

        # Primitive X are the finite "physics" of the experiment.
        # Complete task recipes are NOT pre-specified.
        for p in REGISTRY.values():
            self.store.add(pack_recipe([p.code]), "primitive", p.name)

    # ---------- observations / generic role constraints ----------

    @staticmethod
    def roles_needed(obs: ObservationSignature) -> Tuple[str, ...]:
        roles: List[str] = []
        if obs.has("has_filter"):
            roles.append("filter")
        if obs.has("has_compare"):
            roles.append("compare")
        if obs.has("has_extreme"):
            roles.append("extreme")
        if obs.has("has_aggregate") or obs.has("has_select"):
            roles.append("terminal")
        return tuple(roles)

    @staticmethod
    def structural_key(obs: ObservationSignature) -> str:
        """
        Index key contains structural evidence only, never literal object values.
        This is the first-stage Lāma retrieval address.
        """
        return "|".join(sorted(obs.features))

    @staticmethod
    def feature_role(feature: str) -> str:
        if feature == "has_filter" or feature.startswith("filter_"):
            return "filter"
        if feature == "has_compare" or feature.startswith("compare_"):
            return "compare"
        if feature == "has_extreme" or feature.startswith("extreme"):
            return "extreme"
        if (
            feature in {"has_aggregate", "has_select"}
            or feature.startswith("operation:")
            or feature.startswith("aggregate_field:")
            or feature.startswith("select_field:")
        ):
            return "terminal"
        return ""

    # ---------- primitive routing ----------

    def primitive_score(self, obs: ObservationSignature, p: PrimitiveX) -> float:
        # No task-answer semantic prior.  All primitives inside a needed generic
        # role begin tied. Exact identity must be learned from successful search.
        score = 0.05
        for feature in obs.features:
            score += self.store.primitive_weight(feature, p.code)
        return score

    def primitive_candidates(
        self,
        obs: ObservationSignature,
        role: str,
    ) -> List[Tuple[int, float]]:
        cands = [
            (p.code, self.primitive_score(obs, p))
            for p in REGISTRY.values()
            if p.role == role
        ]
        cands.sort(key=lambda x: x[1], reverse=True)
        return cands[: self.per_role_candidates]

    # ---------- recipe retrieval ----------

    def recipe_score(self, obs: ObservationSignature, lama_x: int) -> float:
        rec = self.store.records[lama_x]
        codes = unpack_recipe(lama_x)

        # Compare stored and newly synthesized recipes on the same scale:
        # sum of member routing scores + an explicit reuse/confidence bonus.
        score = 0.0
        for code in codes:
            p = CODE_TO_PRIMITIVE.get(code)
            if p is not None:
                score += self.primitive_score(obs, p)

        score += 0.60 * rec.confidence
        score += 0.02 * max(-10.0, min(10.0, rec.reward_mean))
        for f in obs.features:
            score += 0.01 * self.store.recipe_weight(f, lama_x)

        score -= 0.005 * len(codes)
        return score

    def known_recipe_candidates(
        self,
        obs: ObservationSignature,
        roles: Sequence[str],
        limit: int = 8,
    ) -> List[Tuple[int, float]]:
        # O(bucket) retrieval instead of O(all Stored X).
        out = []
        needed = tuple(roles)
        route_key = self.structural_key(obs)

        for x in self.store.recipes_for(route_key):
            rec = self.store.records.get(x)
            if rec is None or rec.kind != "recipe":
                continue
            codes = unpack_recipe(x)
            recipe_roles = tuple(
                CODE_TO_PRIMITIVE[c].role
                for c in codes
                if c in CODE_TO_PRIMITIVE
            )
            if recipe_roles != needed:
                continue
            out.append((x, self.recipe_score(obs, x)))

        out.sort(key=lambda item: item[1], reverse=True)
        return out[:limit]

    # ---------- bounded autonomous search ----------

    def synthesize_candidates(
        self,
        query: Mapping[str, Any],
    ) -> List[Tuple[int, float, str]]:
        """
        Build candidate X recipes without knowing the correct answer.

        Search is bounded and role-factored rather than brute-forcing all X/all paths.
        """
        obs = observe(query)
        roles = self.roles_needed(obs)

        candidates: List[Tuple[int, float, str]] = []

        # Reuse first.
        for x, score in self.known_recipe_candidates(obs, roles):
            candidates.append((x, score + 0.75, "stored_recipe"))

        # New compositions from primitive candidates.
        if roles:
            per_role = [self.primitive_candidates(obs, role) for role in roles]
            for combo in product(*per_role):
                codes = tuple(code for code, _ in combo)
                score = sum(s for _, s in combo)
                x = pack_recipe(codes)
                candidates.append((x, score, "synthesized"))

        # Deduplicate and rank.
        best: Dict[int, Tuple[float, str]] = {}
        for x, score, source in candidates:
            old = best.get(x)
            if old is None or score > old[0]:
                best[x] = (score, source)

        ranked = [(x, score, source) for x, (score, source) in best.items()]
        ranked.sort(key=lambda t: t[1], reverse=True)
        return ranked[: self.search_budget]

    # ---------- learning ----------

    @staticmethod
    def reward(
        correct: bool,
        tried: int,
        active_x: int,
        expanded_primitives: int,
        steps: int,
        discovered: bool,
    ) -> float:
        return (
            (10.0 if correct else -10.0)
            + (2.0 if correct and discovered else 0.0)
            - 0.05 * tried
            - 0.02 * active_x
            - 0.01 * expanded_primitives
            - 0.005 * steps
        )

    def update_from_success(
        self,
        obs: ObservationSignature,
        successful_x: int,
        failed_xs: Sequence[int],
    ) -> None:
        good_codes = unpack_recipe(successful_x)

        # Feature -> successful primitive reinforcement.
        for feature in obs.features:
            role = self.feature_role(feature)
            if not role:
                continue
            for code in good_codes:
                p = CODE_TO_PRIMITIVE.get(code)
                if p and p.role == role:
                    delta = self.learning_rate
                    if feature in p.capabilities:
                        delta *= 1.5
                    self.store.add_primitive_weight(feature, code, +delta)

        # Penalize attempted wrong primitive choices in the same role.
        good_by_role = {
            CODE_TO_PRIMITIVE[c].role: c
            for c in good_codes
            if c in CODE_TO_PRIMITIVE
        }
        for bad_x in failed_xs:
            for code in unpack_recipe(bad_x):
                p = CODE_TO_PRIMITIVE.get(code)
                if not p:
                    continue
                good = good_by_role.get(p.role)
                if good is None or good == code:
                    continue
                for feature in obs.features:
                    if self.feature_role(feature) == p.role:
                        self.store.add_primitive_weight(
                            feature, code, -self.learning_rate * 0.35
                        )

        # Feature -> whole discovered recipe reinforcement.
        for feature in obs.features:
            self.store.add_recipe_weight(feature, successful_x, +self.learning_rate * 0.35)

    def learn_episode(
        self,
        world: Mapping[str, Any],
        query: Mapping[str, Any],
        expected: Any,
    ) -> EpisodeResult:
        obs = observe(query)
        candidates = self.synthesize_candidates(query)

        failed: List[int] = []
        successful_x: Optional[int] = None
        successful_result = None
        successful_source = ""
        tried = 0

        for x, _, source in candidates:
            tried += 1
            r = run_lama_x(x, world, query, expected, has_expected=True)
            if r.verified:
                successful_x = x
                successful_result = r
                successful_source = source
                break
            failed.append(x)
            if x in self.store.records and self.store.records[x].kind == "recipe":
                self.store.update(x, False, -10.0)

        if successful_x is None:
            # No structure discovered inside the current finite Primitive/search budget.
            x = candidates[0][0] if candidates else pack_recipe([])
            rr = run_lama_x(x, world, query, expected, has_expected=True)
            rew = self.reward(False, tried, rr.active_x_count, rr.expanded_primitive_count, rr.steps, False)
            return EpisodeResult(
                correct=False,
                value=rr.value,
                expected=expected,
                lama_x=x,
                primitive_codes=rr.primitive_codes,
                reward=rew,
                tried_plans=tried,
                discovered=False,
                promoted=False,
                used_known_recipe=False,
                active_x=rr.active_x_count,
                expanded_primitives=rr.expanded_primitive_count,
                steps=rr.steps,
            )

        discovered = successful_source == "synthesized"
        used_known = successful_source == "stored_recipe"

        self.update_from_success(obs, successful_x, failed)

        trace_count = self.store.note_trace(unpack_recipe(successful_x))
        promoted = False
        if discovered and trace_count >= self.promotion_threshold:
            if successful_x not in self.store.records:
                names = [
                    CODE_TO_PRIMITIVE[c].name
                    for c in unpack_recipe(successful_x)
                    if c in CODE_TO_PRIMITIVE
                ]
                self.store.add(
                    successful_x,
                    kind="recipe",
                    name=" -> ".join(names),
                    route_key=self.structural_key(obs),
                )
                promoted = True

        # If already stored, update its empirical reliability.
        if successful_x in self.store.records:
            rew_tmp = self.reward(
                True, tried,
                successful_result.active_x_count,
                successful_result.expanded_primitive_count,
                successful_result.steps,
                discovered,
            )
            self.store.update(successful_x, True, rew_tmp)

        rew = self.reward(
            True,
            tried,
            successful_result.active_x_count,
            successful_result.expanded_primitive_count,
            successful_result.steps,
            discovered,
        )

        return EpisodeResult(
            correct=True,
            value=successful_result.value,
            expected=expected,
            lama_x=successful_x,
            primitive_codes=successful_result.primitive_codes,
            reward=rew,
            tried_plans=tried,
            discovered=discovered,
            promoted=promoted,
            used_known_recipe=used_known,
            active_x=successful_result.active_x_count,
            expanded_primitives=successful_result.expanded_primitive_count,
            steps=successful_result.steps,
        )

    # ---------- inference ----------

    def choose_x(self, query: Mapping[str, Any]) -> Tuple[int, str, int]:
        """
        No expected answer, no search execution feedback.
        Choose the highest-scored structural candidate once.
        """
        candidates = self.synthesize_candidates(query)
        if not candidates:
            return pack_recipe([]), "empty", 0
        x, _, source = candidates[0]
        return x, source, len(candidates)

    def infer(
        self,
        world: Mapping[str, Any],
        query: Mapping[str, Any],
    ) -> EpisodeResult:
        x, source, candidate_count = self.choose_x(query)
        r = run_lama_x(x, world, query)
        return EpisodeResult(
            correct=False,
            value=r.value,
            expected=None,
            lama_x=x,
            primitive_codes=r.primitive_codes,
            reward=0.0,
            tried_plans=1,
            discovered=(source == "synthesized"),
            promoted=False,
            used_known_recipe=(source == "stored_recipe"),
            active_x=r.active_x_count,
            expanded_primitives=r.expanded_primitive_count,
            steps=r.steps,
        )
