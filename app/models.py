from typing import Optional

from pydantic import BaseModel


class NidData(BaseModel):
    name: Optional[str] = None
    fatherName: Optional[str] = None
    motherName: Optional[str] = None
    dateOfBirth: Optional[str] = None
    nidNumber: Optional[str] = None
    presentAddress: Optional[str] = None
    permanentAddress: Optional[str] = None


class ExtractResponse(BaseModel):
    success: bool
    data: Optional[NidData] = None
    warnings: list[str] = []
    errors: list[str] = []


class HealthResponse(BaseModel):
    status: str = "ok"
