"""
Causal root cause analysis pipeline for industrial machinery.

The stages run in this order:

    data        load causRCA recordings and the expert causal graph
    preprocess  turn change-event logs into a regular table
    discovery   learn the causal graph from data (PCMCI+)
    evaluate    score the learned graph against the expert graph
    scm         fit one equation per variable, so we can simulate changes
    rca         find which variable caused a fault
    explain     interventions and counterfactuals
    runner      runs all of the above end to end

Settings live in settings.py at the project root.
"""
