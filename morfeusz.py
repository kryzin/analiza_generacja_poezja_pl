import morfeusz2


class MorfWrapper:
    def __init__(self):
        self.morf = morfeusz2.Morfeusz()

    def analyze_text(self, text):
        analysis_result = self.morf.analyse(text)
        analyses = []
        for (start_node, end_node, interpretation) in analysis_result:
            token_form, lemma, tag, posp, quals = interpretation

            analyses.append((token_form, lemma, tag))

        return analyses

    # zwraca formy podstawowe, te najbardziej pasujące
    # np. ('Litwo', 'litwa'), ('Litwo', 'Litwa:Sf'), ('Litwo', 'Litwa:Sm1')
    def get_base_forms(self, text):
        analyses = self.analyze_text(text)
        return [(word, base) for word, base, _ in analyses]

    # zwraca tagi morfoloficzne, też wg popularności
    # ('Litwo', 'subst:sg:voc:f'), ('Litwo', 'subst:sg:voc:m1')
    def get_morphology(self, text):
        analyses = self.analyze_text(text)
        return [(word, tags) for word, _, tags in analyses]


if __name__ == "__main__":
    polimorf = MorfWrapper()

    sample_text = """
    Litwo! Ojczyzno moja! ty jesteś jak zdrowie;
    Ile cię trzeba cenić, ten tylko się dowie,
    Kto cię stracił.
    """

    base_forms = polimorf.get_base_forms(sample_text)
    print("Base forms:", base_forms)

    morphology = polimorf.get_morphology(sample_text)
    print("Morphological analysis:", morphology)
