class TokenTracker:
    def __init__(self, name="Session", input_rate=0.00025, output_rate=0.0005):
        self.name = name
        self.input_tokens = 0
        self.output_tokens = 0
        self.input_rate = input_rate
        self.output_rate = output_rate
        self.warning_threshold = 10.0  # USD

    def add_usage(self, usage_metadata):
        # Use real fields provided by the API
        self.input_tokens += usage_metadata.prompt_token_count or 0
        self.output_tokens += usage_metadata.response_token_count or 0

        # Check cost after updating
        if self.estimated_cost() > self.warning_threshold:
            print(
                f"\n⚠️  WARNING: Estimated cost has exceeded ${self.warning_threshold:.2f}!"
                f" Current: ${self.estimated_cost():.2f}\n"
            )

    def total_tokens(self):
        return self.input_tokens + self.output_tokens

    def estimated_cost(self):
        cost_in = self.input_tokens / 1000 * self.input_rate
        cost_out = self.output_tokens / 1000 * self.output_rate
        return round(cost_in + cost_out, 6)

    def summary(self):
        return (
            f"[{self.name}] Input: {self.input_tokens} tokens, "
            f"Output: {self.output_tokens} tokens, "
            f"Total: {self.total_tokens()} tokens, "
            f"Estimated Cost: ${self.estimated_cost():.6f}"
        )
