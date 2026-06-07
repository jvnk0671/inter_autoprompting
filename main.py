from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional

from autoprompting import (
    CoolPromptOptimizer,
    ExampleOptimiser,
    Pipeline,
    PromptomatixOptimizer,
)

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
    ch_limit: int = Field(default=2000, alias="ch_lim")
    uncertainty: int = 20
    target_model: str = "meta-llama/llama-3.3-70b-instruct"
    system_model: str = "meta-llama/llama-3.3-70b-instruct"
    evaluate: bool = False
    translate: bool = False

    pop_size: Optional[int] = 5
    generations: Optional[int] = 3
    eval_samples: Optional[int] = 3

    class Config:
        populate_by_name = True

class OptimizeResponse(BaseModel):
    optimized_prompt: str
    init_tokens: Optional[int] = None
    final_tokens: Optional[int] = None
    init_score: Optional[float] = None 
    final_score: Optional[float] = None 

@app.get("/")
def root():
    return {"status": "ok"}

@app.post("/optimize", response_model=OptimizeResponse)
def optimize(req: OptimizeRequest):
    if not req.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt is empty")

    if req.method == "coolprompt":
        optimizer = CoolPromptOptimizer(target_model=req.target_model, system_model=req.system_model)
    elif req.method == "my_promptomatix":
        optimizer = PromptomatixOptimizer(target_model=req.target_model, system_model=req.system_model, use_custom=True)
    elif req.method == "promptomatix":
        optimizer = PromptomatixOptimizer(target_model=req.target_model, system_model=req.system_model, use_custom=False)
    elif req.method == "evoprompt":
        from autoprompting import EvoPromptOptimizer
        optimizer = EvoPromptOptimizer(
            target_model=req.target_model, 
            system_model=req.system_model,
            pop_size=req.pop_size,
            generations=req.generations,
            eval_samples=req.eval_samples
        )
    elif req.method == "example":
        optimizer = ExampleOptimiser()
    else:
        raise HTTPException(status_code=400, detail=f"Unknown method: {req.method}")

    pipeline = Pipeline(
        optimizer=optimizer, 
        sys_model=req.system_model,
        target_model=req.target_model
    )

    try:
        res = pipeline.run(
            prompt=req.prompt,
            ch_limit=req.ch_limit,
            uncertainty=req.uncertainty,
            evaluate=req.evaluate,
            translate=req.translate
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return OptimizeResponse(
        optimized_prompt=res.optimized_prompt,
        init_tokens=res.init_tokens,
        final_tokens=res.final_tokens,
        init_score=res.init_score,
        final_score=res.final_score
    )