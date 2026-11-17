"""Predict a person's answer using the fixed direct-prediction template."""

from chiron import Chiron


def main():
    model = Chiron.from_server("http://127.0.0.1:8000/v1")
    record_context = "Below are records describing this person."
    observations = [
        "I prefer spending my free time with a few close friends.",
        "At large gatherings, I usually listen rather than start conversations.",
    ]
    questions = [
        'How would this person rate "I am the life of the party"?\n'
        "Option 1: Strongly disagree\nOption 2: Disagree\n"
        "Option 3: Neither agree nor disagree\nOption 4: Agree\n"
        "Option 5: Strongly agree"
    ]
    print(model.predict(
        record_context=record_context,
        observations=observations,
        questions=questions,
    ))

    # Shortened IPIP-FFM evaluation example: two targets sharing one response scale.
    record_context = (
        "Below are self-report questionnaire responses from one respondent on the IPIP-FFM scale."
    )
    observations = [
        "Item: I worry about things | Response: Agree",
        "Item: I start conversations | Response: Agree",
        "Item: I don't mind being the center of attention | Response: Neutral",
        "Item: I take time out for others | Response: Agree",
    ]
    questions = (
        "Use this response scale for every question below:\n"
        "Option 1: Disagree\nOption 2: Slightly disagree\nOption 3: Neutral\n"
        "Option 4: Slightly agree\nOption 5: Agree\n\n"
        "Rate each statement for this person:\n"
        'Q1. "I am not interested in abstract ideas"\n'
        'Q2. "I get irritated easily"'
    )
    print(model.predict(
        record_context=record_context,
        observations=observations,
        questions=questions,
    ))
    model.close()


if __name__ == "__main__":
    main()
