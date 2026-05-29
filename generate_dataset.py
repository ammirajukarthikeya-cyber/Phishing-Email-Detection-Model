"""
Synthetic Phishing & Legitimate Email Dataset Generator
Generates labeled email samples for training the phishing detection model.
"""

import pandas as pd
import numpy as np
import random
import os

# Seed for reproducibility
np.random.seed(42)
random.seed(42)

# ─── Phishing Email Templates ────────────────────────────────────────────────

PHISHING_SUBJECTS = [
    "URGENT: Your account has been compromised",
    "Action Required: Verify your identity immediately",
    "Your account will be suspended in 24 hours",
    "Congratulations! You've won a $1000 gift card",
    "Security Alert: Unusual sign-in activity detected",
    "Your payment failed - update billing info now",
    "Important: Confirm your account details",
    "WARNING: Unauthorized access attempt on your account",
    "Final Notice: Your account will be closed",
    "Exclusive offer: Act now before it expires!",
    "Your package delivery is pending - confirm address",
    "Tax refund notification - claim your refund now",
    "Reset your password immediately",
    "Invoice #INV-39481 attached - payment overdue",
    "You have (1) unread secure message",
]

PHISHING_BODIES = [
    "Dear Customer, We have detected unusual activity on your account. Please click the link below to verify your identity immediately or your account will be permanently suspended. http://{url} This is an automated security measure. Act now to protect your account.",
    "URGENT NOTICE: Your account has been temporarily locked due to suspicious activity. You must verify your credentials within 24 hours to avoid permanent suspension. Click here: http://{url} Failure to act will result in account deletion.",
    "Congratulations! You have been selected as the winner of our monthly prize draw. To claim your $1,000 gift card, click the link below and enter your personal details: http://{url} Offer expires in 48 hours!",
    "Dear User, We noticed a sign-in attempt from an unrecognized device. If this wasn't you, please secure your account immediately by clicking: http://{url} Your security is our top priority.",
    "Your recent payment of $49.99 has failed. To avoid service interruption, please update your billing information immediately at: http://{url} If you do not update within 12 hours, your account will be suspended.",
    "IMPORTANT: Your email storage is almost full (98% used). Click below to upgrade your storage for FREE before your emails are permanently deleted: http://{url} Act now - this offer won't last!",
    "Dear Valued Customer, As part of our security upgrade, we require all users to re-verify their account information. Please complete the verification process here: http://{url} Thank you for your cooperation.",
    "WARNING: We detected multiple unauthorized login attempts on your account from IP address 192.168.{ip1}.{ip2}. Secure your account NOW: http://{url} Ignoring this message may lead to data breach.",
    "Your tax refund of $3,247.00 is ready to be processed. To receive your refund, please verify your banking details at: http://{url} This is a time-sensitive matter from the IRS.",
    "Hi, I'm reaching out regarding an outstanding invoice #INV-{inv}. Payment of $892.50 is overdue. Please review and process payment immediately: http://{url} Late fees will apply after 48 hours.",
    "FINAL WARNING: Your subscription will be cancelled unless you update your payment method. Don't lose access to your account - update now: http://{url} This is your last chance.",
    "Dear Account Holder, We are updating our privacy policy and need you to review and accept the new terms. Click here to continue using your account: http://{url} Non-compliance will result in account restriction.",
    "You have received a secure document from DocuSign. To view and sign the document, click here: http://{url} This document requires your immediate attention and signature.",
    "Alert: Your password will expire in 2 hours. Reset your password now to maintain access to your account: http://{url} If you did not request this change, contact support immediately.",
    "Special Limited Offer: Get 90% off on all products! This exclusive deal is only available for the next 3 hours. Shop now: http://{url} Use code SAVE90 at checkout. Hurry, stocks are limited!",
]

PHISHING_URLS = [
    "192.168.1.{rand}/verify-account",
    "secure-login.account-verify.{rand}.xyz/auth",
    "bit.ly/{rand_str}",
    "tinyurl.com/{rand_str}",
    "www.paypa1.com/signin/{rand_str}",
    "account-security.{rand_str}.ru/update",
    "login.microsoftonline.{rand_str}.tk/oauth",
    "www.amaz0n-support.com/verify/{rand_str}",
    "appleid.apple.com-verify.{rand_str}.cn/auth",
    "docs.google.com.{rand_str}.info/document",
    "secure.chase-verify.{rand_str}.net/login",
    "portal.office365-security.{rand_str}.co/reset",
    "10.0.{rand}.{rand}/phish/login.php",
    "www.bankofamerica-secure.{rand_str}.org/verify",
    "netflix-billing.{rand_str}.top/update-payment",
]

# ─── Legitimate Email Templates ─────────────────────────────────────────────

LEGITIMATE_SUBJECTS = [
    "Meeting scheduled for Thursday at 2 PM",
    "Q3 Project Update - Progress Report",
    "Team lunch this Friday - RSVP",
    "Monthly newsletter - Company updates",
    "Re: Question about the API documentation",
    "Your order has been shipped",
    "Weekly standup notes - Sprint 14",
    "Invitation: Annual company retreat",
    "Feedback requested on design mockups",
    "Happy Birthday from the team!",
    "New blog post: Best practices for code reviews",
    "Reminder: Submit your timesheet by Friday",
    "Job posting: Senior Software Engineer",
    "Your monthly statement is ready",
    "Conference registration confirmation",
]

LEGITIMATE_BODIES = [
    "Hi team, Just a reminder that we have our weekly sync meeting scheduled for Thursday at 2 PM in Conference Room B. Please review the agenda document shared on our project board before the meeting. Looking forward to a productive discussion. Best regards, Sarah",
    "Hello everyone, I'm pleased to share the Q3 progress report. We've completed 87% of our planned deliverables and are on track to meet all major milestones. Key highlights include the successful launch of the new API, completion of database migration, and positive user feedback on the redesigned dashboard. Full report attached.",
    "Hey all! We're organizing a team lunch this Friday at 12:30 PM at Olive Garden. Please RSVP by Wednesday so I can make the reservation. Vegetarian and vegan options are available. Hope to see everyone there! - Mike",
    "Dear Subscriber, Here's your monthly roundup of company news and updates. This month we welcomed 5 new team members, launched our mobile app beta, and announced our partnership with TechCorp. Read more on our blog at https://www.ourcompany.com/blog",
    "Hi David, Thanks for your question about the API documentation. The rate limiting section has been updated to reflect the new thresholds. You can find the latest docs at https://docs.ourcompany.com/api/v2. Let me know if you have any other questions. Cheers, Lisa",
    "Your order #ORD-284719 has been shipped! You can track your package using the tracking number: TRK-8847261. Estimated delivery: 3-5 business days. Track your order at https://www.shipping-company.com/track. Thank you for your purchase!",
    "Team, Here are the key takeaways from today's standup: Frontend - Login page redesign completed. Backend - API endpoints for user profiles deployed to staging. QA - 12 test cases passed, 2 minor bugs logged. Next sprint planning is scheduled for Monday at 10 AM.",
    "You're invited to the Annual Company Retreat! Date: December 15-17. Location: Mountain View Resort. Activities include team-building exercises, workshops, and networking dinners. Please confirm your attendance by November 30th. Register at https://www.ourcompany.com/retreat",
    "Hi team, The design team has completed the initial mockups for the new customer portal. We'd love your feedback before we move to the development phase. Please review the designs in Figma and add your comments by end of week. Link: https://www.figma.com/file/project-portal",
    "Happy Birthday, Alex! Wishing you a wonderful day filled with joy and celebration. The team has put together a little something for you - check the break room at noon! Best wishes from everyone at the office.",
    "New on our engineering blog: Best Practices for Code Reviews. In this post, we cover how to write constructive feedback, common pitfalls to avoid, and tools that can streamline your review process. Read the full article at https://www.ourcompany.com/blog/code-reviews",
    "Reminder: Please submit your timesheet for this week by Friday 5 PM. You can access the timesheet portal at https://hr.ourcompany.com/timesheet. If you have any issues, contact HR at hr@ourcompany.com. Thanks!",
    "We're hiring! We're looking for a Senior Software Engineer to join our platform team. Requirements include 5+ years of experience, proficiency in Python and Go, and experience with distributed systems. Apply at https://careers.ourcompany.com/senior-swe",
    "Your monthly account statement for October is now available. You can view and download it from your dashboard at https://www.ourbank.com/statements. If you have any questions about your statement, please contact customer service.",
    "Thank you for registering for TechConf 2026! Your registration has been confirmed. Event details: Date: March 15-17, Location: San Francisco Convention Center. Your confirmation number is TC-{conf}. View event schedule at https://www.techconf.com/schedule",
]

LEGITIMATE_SENDERS = [
    "sarah.johnson@ourcompany.com",
    "mike.chen@ourcompany.com",
    "newsletter@ourcompany.com",
    "david.smith@ourcompany.com",
    "orders@amazon.com",
    "team@slack.com",
    "hr@ourcompany.com",
    "noreply@github.com",
    "support@figma.com",
    "no-reply@google.com",
]

PHISHING_SENDERS = [
    "security@acc0unt-verify.com",
    "admin@paypa1-support.xyz",
    "noreply@microsoft-security.tk",
    "support@amaz0n-billing.ru",
    "service@apple-id-verify.cn",
    "alert@bank0famerica.net",
    "admin@chase-secure.info",
    "noreply@netfl1x-update.top",
    "security@0ffice365.co",
    "helpdesk@g00gle-alert.org",
]


def _random_string(length=8):
    """Generate a random alphanumeric string."""
    chars = "abcdefghijklmnopqrstuvwxyz0123456789"
    return "".join(random.choice(chars) for _ in range(length))


def _fill_url(url_template):
    """Fill URL template with random values."""
    return (
        url_template
        .replace("{rand}", str(random.randint(1, 254)))
        .replace("{rand_str}", _random_string())
    )


def _fill_body(body_template):
    """Fill body template with random values."""
    url = _fill_url(random.choice(PHISHING_URLS))
    return (
        body_template
        .replace("{url}", url)
        .replace("{ip1}", str(random.randint(1, 254)))
        .replace("{ip2}", str(random.randint(1, 254)))
        .replace("{inv}", str(random.randint(10000, 99999)))
        .replace("{conf}", str(random.randint(100000, 999999)))
    )


def generate_dataset(n_samples=2000, output_path=None):
    """
    Generate a synthetic dataset of phishing and legitimate emails.

    Args:
        n_samples: Total number of email samples to generate.
        output_path: Path to save the CSV file. Defaults to data/emails.csv.

    Returns:
        pd.DataFrame with columns: [subject, body, sender, label]
    """
    if output_path is None:
        output_path = os.path.join(os.path.dirname(__file__), "emails.csv")

    emails = []
    n_phishing = n_samples // 2
    n_legit = n_samples - n_phishing

    # Generate phishing emails
    for _ in range(n_phishing):
        subject = random.choice(PHISHING_SUBJECTS)
        body = _fill_body(random.choice(PHISHING_BODIES))
        sender = random.choice(PHISHING_SENDERS)

        # Add some variation: random extra urgency phrases
        if random.random() > 0.5:
            urgency = random.choice([
                " ACT NOW!", " Don't delay!", " IMMEDIATE ACTION REQUIRED!",
                " Time is running out!", " This is your FINAL warning!",
            ])
            body += urgency

        # Sometimes add extra suspicious URLs
        if random.random() > 0.6:
            extra_url = _fill_url(random.choice(PHISHING_URLS))
            body += f" Alternative link: http://{extra_url}"

        emails.append({
            "subject": subject,
            "body": body,
            "sender": sender,
            "label": 1,  # 1 = Phishing
        })

    # Generate legitimate emails
    for _ in range(n_legit):
        subject = random.choice(LEGITIMATE_SUBJECTS)
        body = random.choice(LEGITIMATE_BODIES).replace(
            "{conf}", str(random.randint(100000, 999999))
        )
        sender = random.choice(LEGITIMATE_SENDERS)

        emails.append({
            "subject": subject,
            "body": body,
            "sender": sender,
            "label": 0,  # 0 = Safe
        })

    # Shuffle the dataset
    random.shuffle(emails)
    df = pd.DataFrame(emails)

    # Save to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[+] Dataset generated: {len(df)} samples ({n_phishing} phishing, {n_legit} legitimate)")
    print(f"[+] Saved to: {output_path}")

    return df


if __name__ == "__main__":
    generate_dataset()
