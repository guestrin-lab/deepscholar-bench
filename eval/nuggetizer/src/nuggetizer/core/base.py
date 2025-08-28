from abc import ABC, abstractmethod
from typing import List, Union, Protocol, runtime_checkable, Awaitable
from .types import Request, Nugget, ScoredNugget, AssignedNugget, AssignedScoredNugget


# Define a protocol for synchronous Nuggetizer
@runtime_checkable
class NuggetizerProtocol(Protocol):
    def create(self, request: Request) -> List[ScoredNugget]: ...

    def assign(
        self,
        query: str,  # Add query parameter to match implementations
        context: str,
        nuggets: List[ScoredNugget],
    ) -> List[AssignedScoredNugget]: ...

    def create_batch(self, requests: List[Request]) -> List[List[ScoredNugget]]: ...

    def assign_batch(
        self,
        queries: List[str],  # Add queries parameter to match implementations
        contexts: List[str],
        nuggets_list: List[List[ScoredNugget]],
    ) -> List[List[AssignedScoredNugget]]: ...


# Define a protocol for asynchronous Nuggetizer
@runtime_checkable
class AsyncNuggetizerProtocol(Protocol):
    def create(self, request: Request) -> Awaitable[List[ScoredNugget]]: ...

    def assign(
        self, query: str, context: str, nuggets: List[ScoredNugget]
    ) -> Awaitable[List[AssignedScoredNugget]]: ...

    def create_batch(
        self, requests: List[Request]
    ) -> Awaitable[List[List[ScoredNugget]]]: ...

    def assign_batch(
        self,
        queries: List[str],
        contexts: List[str],
        nuggets_list: List[List[ScoredNugget]],
    ) -> Awaitable[List[List[AssignedScoredNugget]]]: ...


# Keep the original ABC for backwards compatibility
class BaseNuggetizer(ABC):
    @abstractmethod
    def create(self, request: Request) -> List[ScoredNugget]:
        pass

    @abstractmethod
    def assign(
        self,
        query: str,  # Add query parameter to match implementations
        context: str,
        nuggets: List[ScoredNugget],
    ) -> List[AssignedScoredNugget]:
        pass

    @abstractmethod
    def create_batch(self, requests: List[Request]) -> List[List[ScoredNugget]]:
        pass

    @abstractmethod
    def assign_batch(
        self,
        queries: List[str],  # Add queries parameter to match implementations
        contexts: List[str],
        nuggets_list: List[List[ScoredNugget]],
    ) -> List[List[AssignedScoredNugget]]:
        pass


class BaseNuggetScorer(ABC):
    @abstractmethod
    def score(self, nuggets: List[Nugget]) -> List[ScoredNugget]:
        pass

    @abstractmethod
    def score_batch(self, nuggets_list: List[List[Nugget]]) -> List[List[ScoredNugget]]:
        pass


class BaseNuggetAssigner(ABC):
    @abstractmethod
    def assign(
        self, context: str, nuggets: Union[List[Nugget], List[ScoredNugget]]
    ) -> Union[List[AssignedNugget], List[AssignedScoredNugget]]:
        pass

    @abstractmethod
    def assign_batch(
        self,
        contexts: List[str],
        nuggets_list: Union[List[List[Nugget]], List[List[ScoredNugget]]],
    ) -> Union[List[List[AssignedNugget]], List[List[AssignedScoredNugget]]]:
        pass
