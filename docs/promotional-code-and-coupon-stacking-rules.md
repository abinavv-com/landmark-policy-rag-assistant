# Promotional Code and Coupon Stacking Rules

This document sets out the general hierarchy governing how promotional codes, sale pricing, Landmark Rewards points, the Landmark+ discount, and the student/staff discount interact within a single order. Individual policy documents (Landmark+ Membership FAQ, Student and Staff Discount Program, Sale Event Rules) each describe their own stacking behavior in context; this document consolidates those rules into one reference and resolves the general hierarchy CX should apply when a customer asks whether two discounts can be combined.

## The Core Stacking Principle

Landmark's checkout engine treats discounts as falling into two categories that behave differently: **percentage-off or flat-value discount mechanisms** (promo codes, the Landmark+ 5% discount, the student/staff 10% discount, and sale/markdown pricing itself), and **value-restoring mechanisms** (Landmark Rewards points redemption and gift card or store credit balance). Only one percentage-off or flat-value discount mechanism can be applied per order. Value-restoring mechanisms are never counted against this limit and always apply in addition to whichever single percentage/flat discount is active, since points, gift cards, and store credit are treated as a form of payment rather than a price reduction.

In practice, this means a single order can combine: (1) one sale price or promo code or membership-based percentage discount, plus (2) Landmark Rewards points redemption, plus (3) a gift card or store credit balance used as payment — but not two items from category (1) at once.

## Promo Code Types

Landmark issues two structural types of promotional codes:

- **Single-use codes**, typically issued to an individual customer (a first-order welcome code, a birthday-month voucher, or a goodwill code issued through the Complaint Escalation Process) and tied to that customer's account; the code is consumed on first successful use and cannot be reapplied.
- **Multi-use campaign codes**, published broadly for a promotional period (for example, a seasonal "WELCOME10" style code) and usable by any customer within the stated validity window, though still capped at one redemption per customer account to prevent a single customer from applying the same public code across multiple orders.

All promo codes carry a stated expiry date shown at the time of issuance; codes not redeemed by the expiry date lapse automatically and are not extended or reissued, except where a specific goodwill exception is approved by a Tier 2 or Tier 3 agent under the Complaint Escalation Process.

## Percentage-Discount Hierarchy: What Wins When Two Would Apply

Where a customer's order would otherwise qualify for more than one percentage-off or flat-value mechanism simultaneously, the checkout system applies a single resolution rule: **the largest single discount value applies, and the others are not stacked on top of it.** This resolves the specific cases already described in their respective policy documents:

- The Landmark+ 5% discount does not stack with the student/staff 10% discount; the checkout system applies whichever is larger for that order (as described in the Landmark+ Membership FAQ) — in most cases this means the 10% student/staff discount applies instead of the 5% Landmark+ discount, since 10% exceeds 5% on a like-for-like order value, though the comparison is evaluated per order rather than assumed.
- Neither the Landmark+ discount nor the student/staff discount stacks with an already-marked-down sale price; sale pricing itself is treated as the active discount mechanism for that item, and percentage-discount codes typically exclude sale and clearance stock under their own terms (see the Student and Staff Discount Program's category exclusions).
- A public promo code and the Landmark+ 5% discount cannot both apply to the same order; the customer (or the checkout system automatically, where only one can legally apply) selects the single larger benefit.

## Loyalty Points Always Accrue Regardless of Stacking

Consistent with the standing rule described in the Loyalty Program FAQ, Landmark Rewards points accrue on the post-discount, pre-shipping amount actually paid, regardless of which single percentage-discount mechanism was used to arrive at that final price. A customer who pays using a promo code, the Landmark+ discount, the student/staff discount, or sale pricing still earns points normally on whatever amount was actually charged; points earning is never suspended or reduced beyond this standard post-discount calculation as a consequence of which discount type was applied.

## Interaction With EOSS and Ramadan Campaigns

During EOSS and the Ramadan campaign, the student/staff discount does not apply at all, since campaign markdowns are treated as sale pricing under that program's exclusion terms, as set out in the Sale Event Rules document. The Landmark+ discount similarly does not apply on top of campaign markdown pricing, following the same single-active-discount principle described above; however, points redemption and the 3x points multiplier both continue to apply during these campaigns, since points mechanics are value-restoring rather than percentage-discount mechanisms and are not affected by the campaign's discount-stacking suspension.

## Reporting a Stacking Discrepancy

If a customer reports that a checkout page appeared to apply two percentage discounts simultaneously (a system error rather than an intended benefit), this should be routed as a bug report per the triage approach in the App and Website Troubleshooting FAQ, since the checkout engine is designed to enforce the single-discount rule automatically and any order showing otherwise reflects a technical fault rather than an approved dual-discount policy exception.
