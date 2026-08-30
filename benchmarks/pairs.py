"""Labeled query pairs for the threshold sweep.

Two classes, hand-labeled:

``PARAPHRASES``  same information need, different wording. A cache *should*
                serve the second from the first.
``NEAR_MISSES``  different information need, deliberately similar surface form
                -- usually one swapped entity, threshold, or scope. A cache
                that serves these returns the wrong regulatory answer with
                full confidence.

The near-miss set is the one that matters. Any similarity measure can score
paraphrases highly; the question is whether it also scores these highly.
"""

PARAPHRASES: list[tuple[str, str]] = [
    (
        "How long must KYC records be kept?",
        "What is the retention period for KYC records?",
    ),
    (
        "Do transfers over 500,000 NOK need multi-factor authentication?",
        "Is MFA required for transfers above 500,000 NOK?",
    ),
    (
        "Where must customer data be stored?",
        "What are the data residency requirements for customer data?",
    ),
    (
        "How often must we run an internal audit?",
        "What is the required frequency of internal audits?",
    ),
    (
        "How quickly must we report a data breach?",
        "What is the deadline for notifying authorities of a data breach?",
    ),
    (
        "Who is allowed to approve a high-value transaction?",
        "Which roles can sign off on a high-value transaction?",
    ),
    (
        "What happens if a customer refuses identity verification?",
        "What is the procedure when a customer declines to verify their identity?",
    ),
    (
        "Can we use a third-party provider for identity checks?",
        "Is outsourcing identity verification to a vendor permitted?",
    ),
    (
        "What records must be kept for cross-border payments?",
        "Which documentation is required for international payment transactions?",
    ),
    (
        "How long is a suspicious activity report retained?",
        "What is the storage duration for a filed suspicious activity report?",
    ),
    (
        "Must staff complete compliance training every year?",
        "Is annual compliance training mandatory for employees?",
    ),
    (
        "What is the threshold for reporting a cash transaction?",
        "Above what amount must a cash transaction be reported?",
    ),
    (
        "Can customer data be transferred outside the EEA?",
        "Is it permitted to send customer data to a country outside the EEA?",
    ),
    (
        "What are the penalties for late regulatory filing?",
        "What fines apply if a regulatory filing is submitted after the deadline?",
    ),
    (
        "Who must approve changes to the risk model?",
        "Which body signs off on modifications to the risk model?",
    ),
    (
        "How is a politically exposed person identified?",
        "What is the process for determining whether a customer is a PEP?",
    ),
    (
        "What encryption is required for data at rest?",
        "Which encryption standard must be applied to stored data?",
    ),
    (
        "How long do we keep transaction logs?",
        "What is the retention window for transaction logs?",
    ),
    (
        "When must a customer file be re-reviewed?",
        "At what point does a customer file require periodic re-review?",
    ),
    (
        "Is a written record of the risk assessment required?",
        "Must the risk assessment be documented in writing?",
    ),
    (
        "What triggers enhanced due diligence?",
        "Under what circumstances is enhanced due diligence required?",
    ),
    (
        "Can an account be opened before verification completes?",
        "Is it allowed to open an account while verification is still pending?",
    ),
    (
        "Who owns the incident response process?",
        "Which team is responsible for incident response?",
    ),
    (
        "What is the maximum retention period for marketing data?",
        "How long may marketing data be retained at most?",
    ),
    (
        "Does the policy apply to contractors?",
        "Are contractors covered by this policy?",
    ),
]

NEAR_MISSES: list[tuple[str, str]] = [
    (
        "What is the retention limit for KYC records?",
        "What is the retention limit for AML records?",
    ),
    (
        "How long must KYC records be kept?",
        "How long must tax records be kept?",
    ),
    (
        "Do transfers over 500,000 NOK need multi-factor authentication?",
        "Do transfers over 50,000 NOK need multi-factor authentication?",
    ),
    (
        "Where must customer data be stored?",
        "Where must employee data be stored?",
    ),
    (
        "How quickly must we report a data breach?",
        "How quickly must we report a suspicious transaction?",
    ),
    (
        "What is the retention period for customer records?",
        "What is the deletion deadline for customer records?",
    ),
    (
        "Who is allowed to approve a high-value transaction?",
        "Who is allowed to reverse a high-value transaction?",
    ),
    (
        "Is MFA required for internal transfers?",
        "Is MFA required for external transfers?",
    ),
    (
        "What are the requirements for onboarding a retail customer?",
        "What are the requirements for onboarding a corporate customer?",
    ),
    (
        "How often must we run an internal audit?",
        "How often must we run an external audit?",
    ),
    (
        "Can customer data be transferred outside the EEA?",
        "Can customer data be transferred outside the United States?",
    ),
    (
        "What is the threshold for reporting a cash transaction?",
        "What is the threshold for reporting a wire transaction?",
    ),
    (
        "What encryption is required for data at rest?",
        "What encryption is required for data in transit?",
    ),
    (
        "What triggers enhanced due diligence?",
        "What triggers simplified due diligence?",
    ),
    (
        "How long is a suspicious activity report retained?",
        "How long is an internal audit report retained?",
    ),
    (
        "Must staff complete compliance training every year?",
        "Must staff complete security training every year?",
    ),
    (
        "What are the penalties for late regulatory filing?",
        "What are the penalties for inaccurate regulatory filing?",
    ),
    (
        "Who must approve changes to the risk model?",
        "Who must approve changes to the pricing model?",
    ),
    (
        "Is a written record of the risk assessment required?",
        "Is a written record of the board decision required?",
    ),
    (
        "Can an account be opened before verification completes?",
        "Can an account be closed before verification completes?",
    ),
    (
        "What is the maximum retention period for marketing data?",
        "What is the minimum retention period for marketing data?",
    ),
    (
        "Does the policy apply to contractors?",
        "Does the policy apply to subsidiaries?",
    ),
    (
        "When must a customer file be re-reviewed?",
        "When must a customer file be archived?",
    ),
    (
        "What records must be kept for cross-border payments?",
        "What records must be kept for domestic payments?",
    ),
    (
        "How is a politically exposed person identified?",
        "How is a sanctioned person identified?",
    ),
]
