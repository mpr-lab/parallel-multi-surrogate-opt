flowchart TD

    Start[" Start<br/>Define search space + objective"]:::pink1

    Warm["Warm Start<br/>Sample a few random hyperparameter settings<br/>so the surrogates have initial data"]:::pink2

    Eval0[" Initial Evaluation<br/>Run those starting configurations<br/>and record real objective scores"]:::pink3

    DS[" DataStore<br/><b>Shared optimization history</b><br/>Stores X = tried configurations<br/>and y = observed scores<br/>+ round, source, timing, metadata"]:::pink4

    ORCH[" Orchestrator<br/><b>Controls the optimization loop</b><br/>Reads history → fits models → requests proposals<br/>→ sends batch for evaluation → stores results"]:::pink1

    SA[" Surrogate A<br/><b>Cheap model of the objective</b><br/>Fits on the shared X,y history<br/>Predicts promising unseen configurations<br/>and proposes ranked candidates"]:::pink2

    SB[" Surrogate B<br/><b>Second independent opinion</b><br/>Fits on the same X,y history<br/>May learn different patterns / uncertainty<br/>and proposes its own ranked candidates"]:::pink2

    Merge[" Candidate Merge<br/><b>Build one clean evaluation batch</b><br/>Combine A + B proposals<br/>remove duplicates<br/>refill from next-ranked candidates"]:::pink3

    Pool[" Worker Pool<br/><b>Parallel evaluation layer</b><br/>Assign candidate configurations to workers<br/>so multiple real evaluations can run at once"]:::pink4

    Obj[" Objective Function<br/><b>The real expensive task</b><br/>Example: train a model with those hyperparameters<br/>and return validation accuracy / loss / runtime"]:::pink1

    Results[" Evaluation Results<br/>Real y values + timing metadata<br/>+ which surrogate proposed each candidate"]:::pink3

    Log[" Contribution Logging<br/>Store surrogate_source<br/>candidate_rank_within_source<br/>improved_global_best"]:::pink2

    Stop{"Budget finished?<br/>Rounds / evaluations / time"}:::decision

    Done[" Finish<br/>Best configuration found<br/>+ full history for analysis<br/>+ per-surrogate contribution data"]:::pink4


    Start --> Warm
    Warm --> Eval0
    Eval0 --> DS

    DS --> ORCH

    ORCH -->|"same X,y history"| SA
    ORCH -->|"same X,y history"| SB

    SA -->|"ranked candidate list"| Merge
    SB -->|"ranked candidate list"| Merge

    Merge --> Pool
    Pool --> Obj
    Obj --> Results
    Results --> Log
    Log --> DS

    DS --> Stop
    Stop -->|"No — next round"| ORCH
    Stop -->|"Yes"| Done


    classDef pink1 fill:#fde2ee,stroke:#d86f9b,stroke-width:2px,color:#4d2440;
    classDef pink2 fill:#f9c5d5,stroke:#c85c8e,stroke-width:2px,color:#4d2440;
    classDef pink3 fill:#f7b2cf,stroke:#b84f86,stroke-width:2px,color:#4d2440;
    classDef pink4 fill:#f3a6c8,stroke:#a7447d,stroke-width:2px,color:#4d2440;
    classDef decision fill:#fff0f6,stroke:#d86f9b,stroke-width:2px,color:#4d2440;