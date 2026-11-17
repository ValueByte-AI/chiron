"""Measure psychological constructs from a person's records."""

from chiron import Chiron


def main():
    model = Chiron.from_pretrained("chiron-group/chiron")
    records = [
        "I make a plan before starting a project and keep track of its deadlines.",
        "When a task becomes difficult, I usually keep working until it is finished.",
    ]
    print(model.measure(records, constructs=["conscientiousness"]))
    # Omit constructs to return all 109 scores; model.constructs lists their names.


if __name__ == "__main__":
    main()
