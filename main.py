from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Any

from pipeline import Pipeline
from autoprompting import ExampleOptimiser, PromptomatixOptimizer
from cool_prompt import CoolPromptOptimizer

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class OptimizeRequest(BaseModel):
    prompt: str
    method: str = "example"
    ch_limit: int = 100
    uncertainty: int = 20
    target_model: str = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free"
    system_model: str = "meta-llama/llama-3.3-70b-instruct:free"


class OptimizeResponse(BaseModel):
    optimized_prompt: str
    init_metric: Optional[Any] = None
    final_metric: Optional[Any] = None
    init_tokens: Optional[int] = None
    final_tokens: Optional[int] = None


@app.get("/")
def root():
    return {"status": "ok"}


@app.post("/optimize", response_model=OptimizeResponse)
def optimize(req: OptimizeRequest):
    if not req.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt is empty")

    if req.method == "coolprompt":
        optimizer = CoolPromptOptimizer()
    elif req.method == "promptomatix":
        optimizer = PromptomatixOptimizer()
    elif req.method == "example":
        optimizer = ExampleOptimiser()
    else:
        raise HTTPException(status_code=400, detail=f"Unknown method: {req.method}")

    pipeline = Pipeline(
        optimizer=optimizer, model=req.system_model.replace(":free", "")
    )

    try:
        res = pipeline.run(
            prompt=req.prompt,
            ch_limit=req.ch_limit,
            uncertainty=req.uncertainty,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return OptimizeResponse(
        optimized_prompt=res.optimized_prompt,
        init_metric=res.init_metric,
        final_metric=res.final_metric,
        init_tokens=res.init_tokens,
        final_tokens=res.final_tokens,
    )
