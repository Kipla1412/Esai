from ...util import TemplateFormatter

class Generation:

    def __init__(self, path=None, template=None, **kwargs):
        self.path = path
        self.template = template
        self.kwargs = kwargs

    def __call__(self, text, maxlength, stream, stop, defaultrole,stripthink, **kwargs):
        texts = [text] if isinstance(text, str) or isinstance(text[0], dict) else text

        if self.template:
            formatter = TemplateFormatter()
            texts = [formatter.format(self.template, text=x) if isinstance(x, str) else x for x in texts]

        if defaultrole == "user":
            texts = [[{"role": "user", "content": x}] if isinstance(x, str) else x for x in texts]

        results = self.execute(texts, maxlength, stream, stop, **kwargs)

        if stream:
            return results

        results = [self.clean(texts[x], results) for x, results in enumerate(results)]

        return results[0] if isinstance(text, str) or isinstance(text[0], dict) else results

    def isvision(self):
        return False

    def execute(self, texts, maxlength, stream, stop, **kwargs):
        if stream:
            return self.stream(texts, maxlength, stream, stop, **kwargs)
        return list(self.stream(texts, maxlength, stream, stop, **kwargs))

    def clean(self, prompt, result):
        text = result.replace(prompt, "") if isinstance(prompt, str) else result
        return text.replace("$=", "<=").strip()

    def response(self, result):
        streamed = False
        for chunk in result:
            data = chunk["choices"][0]
            text = data.get("text", data.get("message", data.get("delta")))
            text = text if isinstance(text, str) else text.get("content")

            if text is not None and (streamed or text.strip()):
                yield (text.lstrip() if not streamed else text)
                streamed = True

    def stream(self, texts, maxlength, stream, stop, **kwargs):
        raise NotImplementedError
