class MoETokenDispatcher:
    """
    Simplified MoE Token Dispatcher for single-GPU, no parallel groups.
    All tokens and experts live on the same device, so no communication needed.
    """

    def __init__(self, 
                 config=None):
        self.config = config
        self.shared_experts = None

    def dispatch_preprocess(self, tokens, routing_map, probs):
        # Just return inputs as-is
        return tokens, probs

    def token_dispatch(self, hidden_states, probs):
        # No dispatch needed, everything stays local
        return hidden_states, probs

    def dispatch_postprocess(self, hidden_states, probs):
        # Return tokens and probs directly
        return hidden_states, None, probs

    def combine_preprocess(self, hidden_states):
        # No preprocessing needed
        return hidden_states

    def token_combine(self, hidden_states):
        # Nothing to combine across devices, return as-is
        return hidden_states

    def combine_postprocess(self, hidden_states):
        # No reordering needed
        return hidden_states

    def set_shared_experts(self, shared_experts):
        self.shared_experts = shared_experts
