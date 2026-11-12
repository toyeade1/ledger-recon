"""Synthetic demonstration cases. Answers are separate from imported evidence."""
import argparse
import json
import random
from pathlib import Path
from ledger_recon.generator import generate


def make_bundle(seed: int = 11001):
    rng = random.Random(seed)
    bundle = {'invoices': [], 'transactions': [], 'documents': []}
    expected = {}
    def inv(i, customer, amount):
        bundle['invoices'].append({'id': i, 'customer': customer, 'amount_cents': amount, 'currency': 'USD'})
    def txn(i, customer, amount, reference=''):
        bundle['transactions'].append({'id': i, 'customer': customer, 'amount_cents': amount, 'currency': 'USD', 'reference': reference})
    def doc(i, text):
        bundle['documents'].append({'id': i, 'filename': i + '.txt', 'text': text})
    amount = rng.randrange(30000, 95000)
    inv('INV-EXACT', 'Cedar Analytics', amount)
    txn('PAY-EXACT', 'Cedar Analytics', amount, 'INV-EXACT')
    expected['PAY-EXACT'] = {'status': 'review', 'invoice_ids': ['INV-EXACT'], 'kind': 'exact'}
    inv('INV-CREDIT', 'Northstar Trading', 1000000)
    txn('PAY-CREDIT', 'Northstar Trading', 975000, 'November settlement')
    doc('credit-note', 'Credit note: CN-2026-04\nCustomer: Northstar Trading\nCurrency: USD\nInvoice: INV-CREDIT\nCredit amount (cents): 25000\nApproved return of damaged stock. Keep this note with the remittance record.')
    doc('credit-remittance', 'REMITTANCE transaction_id=PAY-CREDIT; customer=Northstar Trading; currency=USD; allocations=INV-CREDIT:975000')
    expected['PAY-CREDIT'] = {'status': 'review', 'invoice_ids': ['INV-CREDIT'], 'kind': 'credit'}
    inv('INV-PARTIAL', 'Maple Supply', 120000)
    txn('PAY-PARTIAL', 'Maple Supply', 40000, 'First installment')
    doc('partial-advice', 'REMITTANCE transaction_id=PAY-PARTIAL; customer=Maple Supply; currency=USD; allocations=INV-PARTIAL:40000')
    expected['PAY-PARTIAL'] = {'status': 'review', 'invoice_ids': ['INV-PARTIAL'], 'kind': 'partial'}
    inv('INV-BUNDLE-A', 'Harbor Tools', 32000)
    inv('INV-BUNDLE-B', 'Harbor Tools', 18000)
    txn('PAY-BUNDLE', 'Harbor Tools', 50000, 'Consolidated payment')
    doc('bundle-advice', 'REMITTANCE transaction_id=PAY-BUNDLE; customer=Harbor Tools; currency=USD; allocations=INV-BUNDLE-A:32000,INV-BUNDLE-B:18000')
    expected['PAY-BUNDLE'] = {'status': 'review', 'invoice_ids': ['INV-BUNDLE-A', 'INV-BUNDLE-B'], 'kind': 'bundle'}
    inv('INV-AMB-A', 'Summit Parts', 22000)
    inv('INV-AMB-B', 'Summit Parts', 22000)
    txn('PAY-AMBIGUOUS', 'Summit Parts', 22000, 'Thank you')
    expected['PAY-AMBIGUOUS'] = {'status': 'blocked', 'invoice_ids': [], 'kind': 'ambiguous'}
    inv('INV-MISSING', 'Acorn Services', 35000)
    txn('PAY-MISSING', 'Acorn Services', 31000, 'Adjustment per agreement')
    expected['PAY-MISSING'] = {'status': 'blocked', 'invoice_ids': [], 'kind': 'missing'}
    inv('INV-CONFLICT-A', 'Birch Labs', 16000)
    inv('INV-CONFLICT-B', 'Birch Labs', 16000)
    txn('PAY-CONFLICT', 'Birch Labs', 16000, 'Corrected remittance')
    doc('conflict-first', 'REMITTANCE transaction_id=PAY-CONFLICT; customer=Birch Labs; currency=USD; allocations=INV-CONFLICT-A:16000')
    doc('conflict-second', 'REMITTANCE transaction_id=PAY-CONFLICT; customer=Birch Labs; currency=USD; allocations=INV-CONFLICT-B:16000')
    expected['PAY-CONFLICT'] = {'status': 'blocked', 'invoice_ids': [], 'kind': 'conflict'}
    inv('INV-OVER', 'Juniper Studio', 10000)
    txn('PAY-OVERPAY', 'Juniper Studio', 15000, 'INV-OVER')
    expected['PAY-OVERPAY'] = {'status': 'blocked', 'invoice_ids': [], 'kind': 'overpayment'}
    # Seeded base case comes directly from the existing environment generator;
    # gold is not accessed and generated currencies are preserved.
    scenario = generate(seed, 'easy')
    source = scenario.invoices[0]
    cents = round(source.amount * 100)
    bundle['invoices'].append({'id': 'INV-SEEDED', 'customer': source.customer, 'amount_cents': cents, 'currency': source.currency})
    bundle['transactions'].append({'id': 'PAY-SEEDED', 'customer': source.customer, 'amount_cents': cents, 'currency': source.currency, 'reference': 'INV-SEEDED'})
    expected['PAY-SEEDED'] = {'status': 'review', 'invoice_ids': ['INV-SEEDED'], 'kind': 'original_generator_exact'}
    return bundle, expected


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, default=Path('demo-bundle.json'))
    p.add_argument('--seed', type=int, default=11001)
    args = p.parse_args()
    args.out.write_text(json.dumps(make_bundle(args.seed)[0], indent=2) + '\n')
    print(args.out)


if __name__ == '__main__':
    main()
