"""Sample the two company-hierarchy conditions in study OBHDP_10_1b."""

from chiron import Chiron


PARTICIPANT_CONTEXT = (
    'I am a person invited to a lab to participate in a study. In this experiment, I will '
    'read a scenario and answer some questions.'
)

OPTIONS = ['1 — Not at all likely', '2', '3', '4', '5', '6', '7 — Very likely']

QUESTIONS = [
    {"text": 'Lie if it would benefit them to do so', "options": OPTIONS},
    {"text": 'Cheat if it would help them get ahead', "options": OPTIONS},
    {"text": 'Behave in a corrupt manner', "options": OPTIONS},
]

CONDITIONS = [
    {
        "condition": 'A',
        "experimental_setting": (
            'Thank you for your interest in this study! In this study, we will ask you to imagine a '
            'company and then answer some questions about it. Imagine a company where employees work '
            'remotely -- they do not interact with each other over the course of their workday. '
            'Imagine that employees in this company are tasked with completing jobs for customers, '
            'and complete many jobs in a given workday. Imagine that this company is structured very '
            'hierarchically. There is a lot of difference in terms of rank between employees of this '
            'company -- for most employees, there is someone at the company who is a higher rank than '
            'them and someone at the company who is a lower rank than them. That is, there is a high '
            'degree of hierarchy between employees of this company.'
        ),
        "n_respondents": 103,
        "seed": 20261154,
    },
    {
        "condition": 'B',
        "experimental_setting": (
            'Thank you for your interest in this study! In this study, we will ask you to imagine a '
            'company and then answer some questions about it. Imagine a company where employees work '
            'remotely -- they do not interact with each other over the course of their workday. '
            'Imagine that employees in this company are tasked with completing jobs for customers, '
            'and complete many jobs in a given workday. Imagine that this company is not structured '
            'very hierarchically. There is not much difference in terms of rank between employees of '
            'this company -- most employees have the same rank at this company. That is, there is a '
            'low degree of hierarchy between employees of this company.'
        ),
        "n_respondents": 99,
        "seed": 20261155,
    },
]


def main():
    model = Chiron.from_server("http://127.0.0.1:8000/v1")
    for condition in CONDITIONS:
        respondents = model.simulate_population(
            participant_context=PARTICIPANT_CONTEXT,
            experimental_setting=condition["experimental_setting"],
            questions=QUESTIONS,
            n_respondents=condition["n_respondents"],
            seed=condition["seed"],
        )
        print(condition["condition"], respondents)

    model.close()


if __name__ == "__main__":
    main()
