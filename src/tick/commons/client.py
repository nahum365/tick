"""Narrow HTTP client for public-subject passes and signed claim bodies."""

from __future__ import annotations

import base64
from datetime import datetime

import httpx
from nacl.signing import SigningKey

from .canonical import canonical_bytes
from .keys import contributor_id
from .models import (
    ClaimAccepted,
    ClaimBody,
    ClaimDetailResponse,
    CreditsResponse,
    DisputeAccepted,
    DisputeRequest,
    GraphResponse,
    LicenseClass,
    PassResponse,
    ReverifyRequest,
    ReverifyResponse,
    ScreenCriterion,
    ScreenRequest,
    ScreenResponse,
    SignedClaimRequest,
)


class CommonsClientError(RuntimeError):
    """A remote refusal kept the signed public claim from being accepted."""

    def __init__(self, code: str, reason: str) -> None:
        super().__init__(f"{code}: {reason}")
        self.code = code
        self.reason = reason


class CommonsClient:
    """The only box-side transport to the commons, shaped without private state."""

    def __init__(
        self,
        base_url: str,
        key: SigningKey,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if not base_url.strip():
            raise ValueError("a commons URL is required; set COMMONS_URL and try again")
        self._root = base_url.rstrip("/")
        self._key = key
        self._transport = transport

    def pass_for(self, ticker: str, observed_before: datetime | None = None) -> PassResponse:
        """Read one impersonal pass, optionally pinned to what was knowable then."""
        if not ticker.strip():
            raise ValueError("a ticker is required; name a public subject and try again")
        params: dict[str, str] = {}
        if observed_before is not None:
            if observed_before.tzinfo is None or observed_before.utcoffset() is None:
                raise ValueError(
                    "--observed-before needs a timezone; include Z or an explicit offset and retry"
                )
            params["observed_before"] = observed_before.isoformat()
        with httpx.Client(transport=self._transport) as client:
            subjects = client.get(self._root + "/v1/subjects", params={"ticker": ticker.upper()})
            self._raise(subjects)
            rows = subjects.json()["subjects"]
            if not rows:
                raise CommonsClientError(
                    "SUBJECT_NOT_FOUND",
                    f"no commons subject matches {ticker.upper()}; try a different ticker",
                )
            response = client.get(
                self._root + f"/v1/subjects/{rows[0]['subject_id']}/pass", params=params
            )
        self._raise(response)
        return PassResponse.model_validate(response.json())

    def _subject_id(self, ticker: str) -> str:
        if not ticker.strip():
            raise ValueError("a ticker is required; name a public subject and try again")
        with httpx.Client(transport=self._transport) as client:
            response = client.get(self._root + "/v1/subjects", params={"ticker": ticker.upper()})
        self._raise(response)
        rows = response.json()["subjects"]
        if not rows:
            raise CommonsClientError(
                "SUBJECT_NOT_FOUND",
                f"no commons subject matches {ticker.upper()}; try a different ticker",
            )
        return str(rows[0]["subject_id"])

    @staticmethod
    def _observed_params(observed_before: datetime | None) -> dict[str, str]:
        if observed_before is None:
            return {}
        if observed_before.tzinfo is None or observed_before.utcoffset() is None:
            raise ValueError(
                "--observed-before needs a timezone; include Z or an explicit offset and retry"
            )
        return {"observed_before": observed_before.isoformat()}

    def screen(
        self,
        criteria: tuple[ScreenCriterion, ...],
        observed_before: datetime | None,
    ) -> ScreenResponse:
        """Run only caller-supplied criteria against the common release cursor."""
        request = ScreenRequest(criteria=criteria, observed_before=observed_before)
        with httpx.Client(transport=self._transport) as client:
            response = client.post(self._root + "/v1/screens", json=request.model_dump(mode="json"))
        self._raise(response)
        return ScreenResponse.model_validate(response.json())

    def graph_for(
        self,
        ticker: str,
        depth: int,
        observed_before: datetime | None,
    ) -> GraphResponse:
        """Read claim-backed neighbors without sending any box state."""
        if depth not in {1, 2}:
            raise ValueError("graph depth must be 1 or 2; choose one and retry")
        subject_id = self._subject_id(ticker)
        params = self._observed_params(observed_before)
        params["depth"] = str(depth)
        with httpx.Client(transport=self._transport) as client:
            response = client.get(self._root + f"/v1/subjects/{subject_id}/graph", params=params)
        self._raise(response)
        return GraphResponse.model_validate(response.json())

    def credits(self, observed_before: datetime | None) -> CreditsResponse:
        """Read the append-only credit sum for this pseudonymous public key."""
        key = contributor_id(self._key)
        with httpx.Client(transport=self._transport) as client:
            response = client.get(
                self._root + f"/v1/credits/{key}",
                params=self._observed_params(observed_before),
            )
        self._raise(response)
        return CreditsResponse.model_validate(response.json())

    def reverify(self, claim_id: str) -> ReverifyResponse:
        """Sign one claim identity and ask the service to re-run its checked recipe."""
        if not claim_id.strip():
            raise ValueError(
                "a claim identity is required; copy one from a released claim and retry"
            )
        signature = self._key.sign(canonical_bytes({"claim_id": claim_id})).signature
        request = ReverifyRequest(
            contributor_id=contributor_id(self._key),
            signature=base64.urlsafe_b64encode(signature).rstrip(b"=").decode("ascii"),
        )
        with httpx.Client(transport=self._transport) as client:
            response = client.post(
                self._root + f"/v1/claims/{claim_id}/reverify",
                json=request.model_dump(mode="json"),
            )
        self._raise(response)
        return ReverifyResponse.model_validate(response.json())

    def dispute(self, claim_id: str, source_id: str, reason_code: str) -> DisputeAccepted:
        """Submit evidence already stored by the commons against one released claim."""
        request = DisputeRequest(source_id=source_id, reason_code=reason_code)
        with httpx.Client(transport=self._transport) as client:
            response = client.post(
                self._root + f"/v1/claims/{claim_id}/disputes",
                json=request.model_dump(mode="json"),
            )
        self._raise(response)
        return DisputeAccepted.model_validate(response.json())

    def claim(self, claim_id: str, observed_before: datetime | None) -> ClaimDetailResponse:
        """Read one claim and its evidence disputes at the common release cursor."""
        with httpx.Client(transport=self._transport) as client:
            response = client.get(
                self._root + f"/v1/claims/{claim_id}",
                params=self._observed_params(observed_before),
            )
        self._raise(response)
        return ClaimDetailResponse.model_validate(response.json())

    def submit(self, claim_body: ClaimBody) -> ClaimAccepted:
        """Sign and send only the closed public-claim body accepted by the gate."""
        if claim_body.source.license_class is LicenseClass.DISPLAY_ONLY:
            raise CommonsClientError(
                "SOURCE_DISPLAY_ONLY",
                "display-only source data must stay on this box; use a redistributable "
                "source or work locally",
            )
        signature = self._key.sign(canonical_bytes(claim_body)).signature
        request = SignedClaimRequest(
            body=claim_body,
            contributor_id=contributor_id(self._key),
            signature=base64.urlsafe_b64encode(signature).rstrip(b"=").decode("ascii"),
        )
        with httpx.Client(transport=self._transport) as client:
            response = client.post(self._root + "/v1/claims", json=request.model_dump(mode="json"))
        self._raise(response)
        return ClaimAccepted.model_validate(response.json())

    @staticmethod
    def _raise(response: httpx.Response) -> None:
        if response.is_success:
            return
        try:
            payload = response.json()
            code = str(payload["code"])
            reason = str(payload["reason"])
        except (ValueError, KeyError, TypeError):
            code = "COMMONS_UNAVAILABLE"
            reason = "the commons returned an unreadable response; retry later or work locally"
        raise CommonsClientError(code, reason)
