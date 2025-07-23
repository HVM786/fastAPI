from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict
from uuid import uuid4
from datetime import datetime
from sqlalchemy import create_engine, Column, String, DateTime, ForeignKey, Text
from sqlalchemy.dialects.sqlite import JSON
from sqlalchemy.orm import sessionmaker, relationship, declarative_base
from sqlalchemy.exc import IntegrityError
from collections import deque

# ---------- Setup ----------
DATABASE_URL = "sqlite:///./workflow.db"  # Can be swapped with PostgreSQL
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()
app = FastAPI(title="Workflow Definition API")

# ---------- DB Models ----------
class WorkflowDB(Base):
    __tablename__ = "workflow"
    id = Column(String, primary_key=True, index=True)
    name = Column(String, nullable=False)
    description = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)
    steps = relationship("StepDB", back_populates="workflow", cascade="all, delete")

class StepDB(Base):
    __tablename__ = "step"
    id = Column(String, primary_key=True, index=True)
    workflow_id = Column(String, ForeignKey("workflow.id", ondelete="CASCADE"))
    name = Column(String, nullable=False)
    type = Column(String, nullable=False)
    config = Column(JSON)
    workflow = relationship("WorkflowDB", back_populates="steps")
    dependencies = relationship("DependencyDB", back_populates="to_step", cascade="all, delete")

class DependencyDB(Base):
    __tablename__ = "dependency"
    id = Column(String, primary_key=True, index=True)
    workflow_id = Column(String, ForeignKey("workflow.id", ondelete="CASCADE"))
    from_step_id = Column(String, ForeignKey("step.id", ondelete="CASCADE"))
    to_step_id = Column(String, ForeignKey("step.id", ondelete="CASCADE"))
    condition = Column(String, default="success")
    to_step = relationship("StepDB", foreign_keys=[to_step_id], back_populates="dependencies")

Base.metadata.create_all(bind=engine)

# ---------- Pydantic Schemas ----------
class Dependency(BaseModel):
    from_step_id: str
    condition: str = "success"

class Step(BaseModel):
    id: Optional[str] = None
    name: str
    type: str
    config: Dict
    dependencies: List[Dependency] = []

class Workflow(BaseModel):
    id: Optional[str] = None
    name: str
    description: Optional[str]
    steps: List[Step]

# ---------- Utility ----------
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ---------- Helper for Graph Ops ----------
def has_cycle(graph):
    visited, stack = set(), set()

    def dfs(node):
        if node in stack:
            return True
        if node in visited:
            return False
        visited.add(node)
        stack.add(node)
        for neighbor in graph.get(node, []):
            if dfs(neighbor):
                return True
        stack.remove(node)
        return False

    return any(dfs(node) for node in graph)

def topological_sort(graph):
    indegree = {node: 0 for node in graph}
    for node in graph:
        for dep in graph[node]:
            indegree[dep] += 1

    q = deque([n for n in graph if indegree[n] == 0])
    order = []

    while q:
        node = q.popleft()
        order.append(node)
        for dep in graph[node]:
            indegree[dep] -= 1
            if indegree[dep] == 0:
                q.append(dep)

    if len(order) != len(graph):
        raise ValueError("Cycle detected")
    return order

# ---------- API Endpoints ----------

@app.post("/workflows", response_model=Workflow)
def create_workflow(workflow: Workflow):
    db = SessionLocal()
    try:
        wf_id = str(uuid4())
        wf_db = WorkflowDB(id=wf_id, name=workflow.name, description=workflow.description)
        db.add(wf_db)
        db.flush()  # Get wf persisted

        # Add steps
        step_id_map = {}
        for step in workflow.steps:
            sid = step.id or str(uuid4())
            step_id_map[step.name] = sid
            step_db = StepDB(id=sid, workflow_id=wf_id, name=step.name, type=step.type, config=step.config)
            db.add(step_db)
            db.flush()

        # Add dependencies
        for step in workflow.steps:
            step_db_id = step_id_map[step.name]
            for dep in step.dependencies:
                dep_db = DependencyDB(
                    id=str(uuid4()),
                    workflow_id=wf_id,
                    from_step_id=dep.from_step_id,
                    to_step_id=step_db_id,
                    condition=dep.condition
                )
                if dep.from_step_id == step_db_id:
                    raise HTTPException(400, f"Step {step_db_id} cannot depend on itself")
                db.add(dep_db)

        db.commit()

        return Workflow(
            id=wf_id,
            name=workflow.name,
            description=workflow.description,
            steps=workflow.steps
        )
    except IntegrityError:
        db.rollback()
        raise HTTPException(500, "Database error")
    finally:
        db.close()

@app.get("/workflows/{workflow_id}/steps")
def list_steps(workflow_id: str):
    db = SessionLocal()
    steps = db.query(StepDB).filter_by(workflow_id=workflow_id).all()
    deps = db.query(DependencyDB).filter_by(workflow_id=workflow_id).all()

    dep_map = {}
    for dep in deps:
        dep_map.setdefault(dep.to_step_id, []).append(dep.from_step_id)

    return [
        {
            "step_id": s.id,
            "name": s.name,
            "prerequisites": dep_map.get(s.id, [])
        }
        for s in steps
    ]

@app.post("/workflows/{workflow_id}/validate")
def validate_workflow(workflow_id: str):
    db = SessionLocal()
    steps = db.query(StepDB).filter_by(workflow_id=workflow_id).all()
    deps = db.query(DependencyDB).filter_by(workflow_id=workflow_id).all()

    graph = {s.id: [] for s in steps}
    for dep in deps:
        if dep.from_step_id == dep.to_step_id:
            raise HTTPException(400, "Self dependency detected")
        graph[dep.from_step_id].append(dep.to_step_id)

    if has_cycle(graph):
        raise HTTPException(400, "Circular dependency detected!")
    return {"message": "Workflow is valid"}

@app.get("/workflows/{workflow_id}/execution-order")
def execution_order(workflow_id: str):
    db = SessionLocal()
    steps = db.query(StepDB).filter_by(workflow_id=workflow_id).all()
    deps = db.query(DependencyDB).filter_by(workflow_id=workflow_id).all()

    graph = {s.id: [] for s in steps}
    for dep in deps:
        graph[dep.from_step_id].append(dep.to_step_id)

    try:
        order = topological_sort(graph)
        return {"execution_order": order}
    except ValueError:
        raise HTTPException(400, "Cycle detected, cannot determine execution order")
