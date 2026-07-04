# App and Website Troubleshooting FAQ

This document covers common technical issues on Landmark Group's apps and websites (Lifestyle, Max, Splash, Home Centre, Centrepoint, Babyshop), basic troubleshooting steps, and guidance on when to route a customer to technical support versus standard customer service.

## Payment Failing at Checkout

The most commonly reported technical issue is a payment that fails or hangs at the final checkout step. Before escalating, customers should be guided through these steps in order: (1) confirm the card has sufficient balance and has not hit a daily online transaction limit set by the issuing bank, (2) retry using a different payment method (an alternate card, Apple Pay/Google Pay, or cash on delivery where available) to isolate whether the issue is card-specific or checkout-wide, (3) clear the app cache or browser cookies and retry, since a stale session token is a frequent cause of repeated failures, and (4) check whether the bank has sent an SMS/OTP prompt that the customer may have missed, since 3D Secure verification timeouts after 90 seconds and silently fails the transaction without a clear error message in some cases.

If a payment fails but the customer's bank statement shows a debit, this is treated as a payment reconciliation issue, not a bug report: CX should confirm the order was not created in the system, and the amount is auto-reversed by the payment gateway within 5-7 business days per standard card network rules. This is communicated to the customer proactively, since gateway-side holds are common and resolve without manual intervention in the large majority of cases.

## App Crashes on Order History

A known intermittent issue affects the Order History screen crashing or showing a blank screen, most often on devices running an app version more than two major releases behind the current App Store/Play Store version. Basic troubleshooting: (1) force-close and reopen the app, (2) check for and install any pending app update, (3) if the crash persists after updating, uninstall and reinstall the app, since a corrupted local cache of order data is the most common root cause and a fresh install clears it. Order history and order status remain fully accessible via the website even when the app is affected, so this is offered as an immediate workaround while troubleshooting proceeds.

## OTP Not Received

When a customer reports not receiving a one-time password for login or checkout verification, first confirm the mobile number on file is correct and that the customer is checking the right delivery channel (SMS vs. WhatsApp-based OTP, which is offered as an alternate delivery method in UAE and KSA). OTPs typically arrive within 60 seconds; if not received within 2 minutes, the customer should use the "Resend OTP" option, which is rate-limited to 3 requests within a 10-minute window to prevent abuse. Persistent non-delivery, especially on a specific carrier, is usually a carrier-side SMS filtering issue rather than a Landmark platform fault, and switching to the WhatsApp OTP option (where enabled) resolves the large majority of these cases without further escalation.

## Reporting a Bug vs. a Policy Question

Not every complaint routed to CX is a technical bug, and correct routing matters for resolution speed. A bug report is anything where the app or website is not behaving as designed — a crash, a broken button, an incorrect price displayed due to a caching error, or a payment gateway timeout. A policy question is anything where the platform is functioning correctly but the customer disagrees with or is confused about a stated policy — for example, a sale item's return window, warranty exclusions, or shipping fees. CX agents should ask "is something broken, or does this look intentional but unclear/unwelcome" as the first triage question: bug reports go to the technical support queue with a screenshot or screen recording where possible, while policy questions are handled directly by CX using the relevant policy document without a technical ticket.

## Supported Devices and Browsers

The Landmark apps are supported on iOS 15 and above and Android 10 and above; devices on older OS versions can still browse but may experience degraded performance on checkout and image-heavy category pages, since older OS versions are not covered by active QA testing. On the web, the storefronts are fully supported on the current and prior major version of Chrome, Safari, Edge, and Firefox; Internet Explorer is not supported in any version and displays a browser-upgrade banner rather than the storefront.

## Known Outage Communication

During a confirmed platform-wide outage (payment gateway down, site unreachable, or app-wide crash affecting a large share of sessions), status updates are posted to the brand's official Instagram/X account and, for extended outages beyond 30 minutes, an in-app or website banner is activated once the platform is partially restored. CX agents should check the internal status dashboard before telling a customer an issue is isolated to their device, since a wrongly-attributed individual troubleshooting session during a platform-wide outage is a common source of customer frustration.
