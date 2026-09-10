from peft import get_peft_model, LoraConfig, TaskType

def apply_lora(model, args):
    lora_config = LoraConfig(task_type=TaskType.CAUSAL_LM, **args["train_args"]["lora_args"])
    model.llm = get_peft_model(model.llm, lora_config)
    model.llm.print_trainable_parameters()
