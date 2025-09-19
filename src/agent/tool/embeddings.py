from smolagents import Tool

from ...embeddings import Embeddings

class EmbeddingsTool(Tool):
    def _init_(self, config):
        """
        config: embeddings tool configuration
        self.description: explains the whole tool, what it does?
        inputs["query"]["description"]: function level docstring for each parameter
        """
        
        self.name = config["name"]
        self.description = f"""{config['description']}. Results are returned as a list of dict elements.
Each result has keys 'id', 'text', 'score'."""
        
        self.inputs = {"query": {"type": "string", "description": "The search query to perform."}}
        self.output_type = "any"

        self.embeddings = self.load(config)

        super()._init_()

    def forward(self, query):
        """
        each result is a dict with {id, text, scoso re}
        """
        return self.embeddings.search(query, 5) #return the top 5 results
    
    def load(self, config):
        if "target" in config:
            return config["target"]
        
        embeddings = Embeddings()
        embeddings.load(**config)

        return embeddings