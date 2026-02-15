from typing import Literal, List, Optional
from pydantic import BaseModel, Field

# Supported Legal Roles for Metadata (Not classification of content, but document identity)
DocTypeEnum = Literal[
    "صحيفة دعوى",      # Statement of Claim
    "مذكرة دفاع",      # Defense Memo
    "مذكرة رد",        # Reply Memo
    "حافظة مستندات",   # Evidence Portfolio
    "محضر جلسة",       # Hearing Minutes
    "حكم تمهيدي",      # Preliminary Judgment
    "غير محدد"         # Unknown
]

PartyEnum = Literal[
    "المدعي",          # Plaintiff
    "المدعى عليه",     # Defendant
    "النيابة",         # Prosecution
    "المحكمة",         # The Court
    "خبير",            # Expert
    "غير محدد"         # Unknown/Neutral
]

class NormalizedChunk(BaseModel):
    """Atomic unit of text for the pipeline."""
    chunk_id: str = Field(..., description="Unique hash of the content")
    doc_id: str = Field(..., description="Parent document ID")
    
    # Traceability
    page_number: int = Field(..., description="1-based page number")
    paragraph_number: int = Field(..., description="1-based paragraph index on the page")
    
    # Content
    clean_text: str = Field(..., description="Text content with headers/stamps removed")
    
    # Context Metadata (Extracted in Node 0)
    doc_type: DocTypeEnum = Field(..., description="Type of the parent document")
    party: PartyEnum = Field(..., description="The party submitting/authoring this document")

class Node0Output(BaseModel):
    chunks: List[NormalizedChunk]

class DocumentMetadata(BaseModel):
    doc_type: DocTypeEnum = Field(description="The legal type of the document")
    party: PartyEnum = Field(description="The party presenting the document (e.g., Plaintiff, Defendant)")
