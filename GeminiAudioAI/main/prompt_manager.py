import os
import sys
from langsmith import Client


class LLangC_Prompt_Manager:
    """
    LLangC_Prompt_Manager manages the retrieval and handling of prompt templates from the LangChain platform.

    Attributes:
        version (int): The version of the prompt to use. Defaults to 2.
        tools: Placeholder for tools, not initialized by default.
        model: Placeholder for model, not initialized by default.
        prompt_name (str): The name of the prompt to fetch from LangChain.
        client: The LangChain client instance for API access.
        prompt: The loaded prompt object.
        prompt_template (str): The template string of the loaded prompt.
        prompt_identifier (str): The unique identifier for the prompt, including version and commit hash.

    Methods:
        get_llang_chain_access():
            Initializes the LangChain client and fetches the prompt by name.
            Raises:
                ValueError: If prompt_name is not set before fetching the prompt.
            Returns:
                bool: True if the prompt is successfully fetched.

        load_prompt_name():
            Sets the prompt_name based on the version.
            Initializes the LangChain client if not already set.
            Retrieves the prompt object and its last commit hash.
            Constructs the prompt_identifier using the prompt name and commit hash.
            Loads the prompt and its template.
            Prints status messages about the prompt loading process.
    """

    def __init__(self, version=2) -> None:
        self.version = version
        self.tools = None
        self.model = None
        self.prompt_name = None
        self.client = None
        self.prompt = None
        self.prompt_template = None
        self.prompt_identifier = None

    def get_llang_chain_access(self):
        """
        Initializes a client using the LANGSMITH_API_KEY environment variable and fetches a prompt by name.

        Raises:
            ValueError: If self.prompt_name is not set before calling this method.

        Returns:
            bool: True if the prompt was successfully fetched and assigned.
        """
        self.client = Client(api_key=os.environ.get("LANGSMITH_API_KEY"))
        # self.prompt_name should be set first!
        if not self.prompt_name:
            raise ValueError("Prompt name must be set before fetching prompt.")
        self.prompt = self.client.pull_prompt(self.prompt_name, include_model=False)

        return True

    def load_prompt_name(self):
        """
        Loads and sets the prompt name and template based on the current version.

        This method performs the following steps:
        1. Maps the current version to a corresponding prompt name.
        2. Ensures the prompt name starts with "gemi".
        3. Initializes the client if it is not already set, using the LANGSMITH_API_KEY environment variable.
        4. Retrieves the prompt object from the client and extracts its dictionary representation.
        5. Obtains the last commit hash of the prompt, if available, and prints it.
        6. Constructs a prompt identifier using the prompt name and last commit hash.
        7. Pulls the prompt using the constructed identifier and sets the prompt and its template as instance attributes.
        8. Prints the loaded prompt template.

        Raises:
            KeyError: If the current version is not present in the prompt_version_map.
            AssertionError: If the prompt name does not start with "gemi".
        """
        prompt_version_map = {
            1: "geminiaudioai_prompt_version_1",
            2: "geminaudioai_prompt_v2",
        }
        self.prompt_name = prompt_version_map[self.version]
        assert self.prompt_name.startswith("gemi")


        # Use self.client here!
        if not self.client:
            self.client = Client(api_key=os.environ.get("LANGSMITH_API_KEY"))
        # This returns a Prompt object; use .dict() to access as dict
        prompt_obj = self.client.get_prompt(prompt_identifier=self.prompt_name)
        # Ensure prompt_obj is
        # not None before calling dict()
        if prompt_obj is None:
            raise ValueError(
                f"Prompt object for \
                             '{self.prompt_name}' not found."
            )
        prompt_dict = prompt_obj.dict()

        prompt_last_comm_hash = prompt_dict.get("last_commit_hash", None)
        if prompt_last_comm_hash:

            self.prompt_identifier = self.prompt_name + ":" + prompt_last_comm_hash
            prompt_pulled = self.client.pull_prompt(
                prompt_identifier=self.prompt_identifier, include_model=False
            ).get_prompts()[0][0]
            self.prompt = prompt_pulled
            self.prompt_template = prompt_pulled.prompt.template
            print("Prompt Template loaded with name", self.prompt_name)

        else:
            print("No prompt last commit hash found from LangChain.")
            print("...Exiting...")
            sys.exit(0)


# Usage example
if __name__ == "__main__":
    pm = LLangC_Prompt_Manager(version=2)
    pm.load_prompt_name()
    pm.get_llang_chain_access()
    # Now you can use pm.prompt in your application
