from typing import List, Literal, Optional
from pydantic import BaseModel, Field
import uuid


class AgentCapabilities(BaseModel):
    streaming: bool = False
    pushNotifications: bool = False
    stateTransitionHistory: bool = False


class AgentSkill(BaseModel):
    id: str
    name: str
    description: str
    examples: List[str] = Field(default_factory=list)


class AgentCard(BaseModel):
    name: str
    description: str
    url: str
    version: str = "1.0.0"
    capabilities: AgentCapabilities = Field(default_factory=AgentCapabilities)
    skills: List[AgentSkill] = Field(default_factory=list)


class A2AMessage(BaseModel):
    role: Literal["user", "agent"]
    content: str


class A2ATask(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    messages: List[A2AMessage]


class A2ATaskResult(BaseModel):
    id: str
    status: Literal["completed", "failed", "input-required"] = "completed"
    output: List[A2AMessage] = Field(default_factory=list)
    error: Optional[str] = None
