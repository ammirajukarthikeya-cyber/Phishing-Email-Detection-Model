"""
Phishing Email Prediction Script

Load the trained model and predict whether new emails are Phishing or Safe.
Run after training with: python predict.py
"""

import os
import sys
import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils import print_banner, print_section, print_metric, COLORS


# --- Sample Emails for Prediction ---------------------------------------------

SAMPLE_EMAILS = [
    {
        "subject": "URGENT: Your account has been compromised - act now!",
        "body": (
            "Dear Customer, We have detected unauthorized access to your account. "
            "Your account will be suspended within 24 hours unless you verify your "
            "identity immediately. Click here to secure your account: "
            "http://192.168.1.45/verify-account?user=target123 "
            "FAILURE TO ACT WILL RESULT IN PERMANENT ACCOUNT DELETION!"
        ),
    },
    {
        "subject": "Team standup notes - Sprint 22",
        "body": (
            "Hi team, Here are the notes from today's standup meeting. "
            "Frontend: Completed the dashboard redesign, merged PR #487. "
            "Backend: API performance optimization deployed to staging. "
            "QA: All regression tests passing. Next planning session is Monday at 10 AM. "
            "Have a great weekend! - Sarah"
        ),
    },
    {
        "subject": "Congratulations! You've won a $5000 prize!",
        "body": (
            "CONGRATULATIONS!!! You have been randomly selected as the WINNER of our "
            "annual sweepstakes! To claim your $5,000 cash prize, click the link below "
            "and enter your bank details: http://bit.ly/pr1ze-claim "
            "This offer expires in 48 hours! ACT NOW! Don't miss out on this "
            "once-in-a-lifetime opportunity! http://tinyurl.com/w1nner-verify"
        ),
    },
    {
        "subject": "Your monthly account statement is ready",
        "body": (
            "Dear Customer, Your October statement is now available for viewing. "
            "You can access your statement by logging into your account at "
            "https://www.ourbank.com/statements. If you have any questions, "
            "please contact our support team at support@ourbank.com. "
            "Thank you for banking with us."
        ),
    },
    {
        "subject": "Security Alert: Reset your password immediately",
        "body": (
            "WARNING! Multiple failed login attempts detected on your account from "
            "IP address 10.0.45.123. Your account security is at risk! "
            "Reset your password NOW before unauthorized users gain access: "
            "http://secure-login.account-verify.x7k2m9.xyz/auth "
            "If you do not act within 2 hours, your account will be PERMANENTLY LOCKED. "
            "This is your FINAL WARNING!"
        ),
    },
    {
        "subject": "Re: API documentation question",
        "body": (
            "Hi David, Thanks for reaching out about the API rate limits. "
            "We recently updated the documentation to reflect the new thresholds. "
            "The base tier now allows 1000 requests/minute, up from 500. "
            "You can find the full docs at https://docs.ourcompany.com/api/v2/rate-limits. "
            "Let me know if you need anything else. Best, Lisa"
        ),
    },
]


def predict_emails(emails, model_path="outputs/phishing_model.pkl", extractor_path="outputs/feature_extractor.pkl"):
    """
    Predict whether emails are phishing or safe.

    Args:
        emails: List of dicts with 'subject' and 'body' keys.
        model_path: Path to the saved model.
        extractor_path: Path to the saved feature extractor.
    """
    # Check if model exists
    if not os.path.exists(model_path) or not os.path.exists(extractor_path):
        print(f"\n  {COLORS['red']}ERROR: Model not found! Run 'python main.py' first to train the model.{COLORS['end']}")
        return

    # Load model and extractor
    print_section("Loading Model", "[LOAD]")
    model = joblib.load(model_path)
    extractor = joblib.load(extractor_path)
    print("  Model and feature extractor loaded successfully!")

    # Predict
    print_section("Prediction Results", "[PREDICT]")
    print()

    for i, email in enumerate(emails, 1):
        text = email["subject"] + " " + email["body"]
        features = extractor.transform([text])

        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]

        label = "PHISHING" if prediction == 1 else "SAFE"
        confidence = probabilities[prediction] * 100
        color = COLORS["red"] if prediction == 1 else COLORS["green"]

        print(f"  {COLORS['bold']}Email {i}:{COLORS['end']}")
        if len(email['subject']) > 60:
            print(f"  Subject: {email['subject'][:60]}...")
        else:
            print(f"  Subject: {email['subject']}")
        print(f"  Result:  {color}{COLORS['bold']}{label}{COLORS['end']}")
        print(f"  Confidence: {color}{confidence:.1f}%{COLORS['end']}")

        # Show probability breakdown
        print(f"  Probability -> Safe: {probabilities[0]:.3f} | Phishing: {probabilities[1]:.3f}")
        print(f"  {'-' * 58}")


def main():
    """Run predictions on sample emails."""
    print_banner()
    predict_emails(SAMPLE_EMAILS)

    print(f"\n{COLORS['cyan']}{COLORS['bold']}{'=' * 62}")
    print(f"  DONE! Prediction complete for {len(SAMPLE_EMAILS)} emails")
    print(f"  TIP: Modify SAMPLE_EMAILS in predict.py to test your own emails")
    print(f"{'=' * 62}{COLORS['end']}\n")


if __name__ == "__main__":
    main()
