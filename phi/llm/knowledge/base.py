from typing import List, Iterator

from pydantic import BaseModel, ConfigDict
from phi.document import Document


class KnowledgeBase(BaseModel):
    """Base class for managing knowledge base"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def get_documents(self) -> Iterator[Document]:
        """Return all documents in the knowledge base"""
        raise NotImplementedError

    def search(self, query: str) -> List[Document]:
        """Return all relevant documents matching the query"""
        raise NotImplementedError

    def load_knowledge_base(self) -> bool:
        """Load the knowledge base to vector db"""
        raise NotImplementedError