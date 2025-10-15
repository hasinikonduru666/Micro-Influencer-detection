#!/usr/bin/env python3
"""
micro_influencer_detection.py
Author: Hasini Konduru
Department of Data Science, Vignan's Institute of Management and Technology for Women
Email: hasinikonduru272@gmail.com

Purpose:
  - Read social interaction data (tweets.csv or sample)
  - Aggregate per user
  - Compute engagement and activity features
  - Compute a normalized Micro-Influencer Score (MIS)
  - Output ranked CSV of micro-influencers (no network visuals)

Usage:
  python micro_influencer_detection.py --input tweets.csv --output micro_influencers_ranked.csv --topk 20
"""

import argparse
import os
import math
import pandas as pd
import numpy as np

# ---------------------------
# Helper functions
# ---------------------------
def safe_div(a, b, eps=1e-9):
    return a / (b + eps)

def minmax_scale(series):
    mn = series.min()
    mx = series.max()
    if mx - mn < 1e-9:
        return series.apply(lambda x: 0.5)  # all same -> middle
    return (series - mn) / (mx - mn)

# ---------------------------
# Core logic
# ---------------------------
def load_or_create_sample(csv_path=None):
    if csv_path and os.path.exists(csv_path):
        print(f"[INFO] Loading dataset from: {csv_path}")
        df = pd.read_csv(csv_path)
        print(f"[INFO] Loaded {len(df)} rows.")
        return df
    else:
        print("[WARN] No input CSV found or provided. Generating sample dataset.")
        data = {
            'user_id': ['U1','U2','U3','U4','U5','U6','U7','U8'],
            'username': ['alice','bob','charlie','diana','eve','frank','gina','harry'],
            'followers': [1200, 5000, 800, 3000, 1500, 450, 7000, 950],
            'retweets': [50, 20, 70, 10, 30, 15, 5, 40],
            'replies': [20, 10, 25, 5, 15, 4, 2, 12],
            'mentions': [10, 5, 12, 3, 8, 1, 0, 6],
            # each row is considered one post for the sample; real CSV may have many rows per user
            'post_id': ['p1','p2','p3','p4','p5','p6','p7','p8']
        }
        df = pd.DataFrame(data)
        return df

def aggregate_user_stats(df):
    # Ensure necessary columns exist
    cols = df.columns.tolist()
    # Columns we expect: user_id, username, followers, retweets, replies, mentions, post_id
    # Fill missing columns with zeros/defaults
    for c in ['user_id','username','followers','retweets','replies','mentions','post_id']:
        if c not in df.columns:
            if c == 'followers':
                df[c] = 0
            else:
                df[c] = ""
    # If multiple rows per user: aggregate sums and follower = max (best estimate)
    agg = df.groupby('username').agg(
        user_id = ('user_id', lambda x: x.dropna().astype(str).iloc[0] if len(x.dropna())>0 else ""),
        followers = ('followers', lambda x: int(max(x.dropna().astype(float)) if len(x.dropna())>0 else 0)),
        total_retweets = ('retweets', lambda x: int(x.fillna(0).astype(float).sum())),
        total_replies = ('replies', lambda x: int(x.fillna(0).astype(float).sum())),
        total_mentions = ('mentions', lambda x: int(x.fillna(0).astype(float).sum())),
        total_posts = ('post_id', lambda x: int(x.replace("", np.nan).dropna().shape[0]) if 'post_id' in df.columns else int(x.shape[0]))
    ).reset_index()
    return agg

def compute_features_and_mis(agg_df, weights=None):
    # Basic features
    agg_df['engagement_raw'] = (agg_df['total_retweets'] + agg_df['total_replies'] + agg_df['total_mentions']).astype(float)
    agg_df['followers'] = agg_df['followers'].astype(float).fillna(0.0)
    agg_df['posts'] = agg_df['total_posts'].astype(float).fillna(0.0)

    # Engagement normalized by followers (avoid divide by 0)
    agg_df['engagement_per_follower'] = agg_df.apply(
        lambda r: safe_div(r['engagement_raw'], r['followers']), axis=1
    )

    # Activity feature: posts (could be zero)
    agg_df['activity'] = agg_df['posts']

    # Inverse followers: users with smaller audience get higher value (micro-influencer tendency)
    agg_df['inv_followers'] = agg_df['followers'].apply(lambda x: safe_div(1.0, (x+1.0)))

    # Min-max scale the three components into [0,1] to combine fairly
    agg_df['eng_norm'] = minmax_scale(agg_df['engagement_per_follower'])
    agg_df['act_norm'] = minmax_scale(agg_df['activity'])
    agg_df['invf_norm'] = minmax_scale(agg_df['inv_followers'])

    # Default weights if not provided
    if weights is None:
        weights = {'eng': 0.6, 'act': 0.2, 'invf': 0.2}

    # Micro-Influencer Score (MIS): weighted sum
    agg_df['MIS'] = (
        weights['eng'] * agg_df['eng_norm'] +
        weights['act'] * agg_df['act_norm'] +
        weights['invf'] * agg_df['invf_norm']
    )

    # Additional helpful fields
    agg_df['engagement_raw'] = agg_df['engagement_raw'].astype(int)
    agg_df['followers'] = agg_df['followers'].astype(int)
    agg_df['total_posts'] = agg_df['posts'].astype(int)

    return agg_df

def save_and_print_results(result_df, output_path, topk=20):
    result_sorted = result_df.sort_values(by='MIS', ascending=False).reset_index(drop=True)
    # Select useful columns
    cols = ['username','user_id','followers','total_posts','total_retweets','total_replies','total_mentions','engagement_per_follower','MIS']
    for c in cols:
        if c not in result_sorted.columns:
            result_sorted[c] = ""
    result_sorted.to_csv(output_path, index=False)
    print(f"\n[INFO] Results saved to: {output_path}")
    print(f"\nTop {min(topk, len(result_sorted))} Micro-Influencers:\n")
    print(result_sorted[cols].head(topk).to_string(index=False))

# ---------------------------
# Command-line interface
# ---------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Micro-Influencer Detection (non-graph version)")
    parser.add_argument('--input', '-i', type=str, default='tweets.csv', help='Input CSV file path (default: tweets.csv).')
    parser.add_argument('--output', '-o', type=str, default='micro_influencers_ranked.csv', help='Output CSV path.')
    parser.add_argument('--topk', '-k', type=int, default=20, help='Number of top users to print.')
    parser.add_argument('--weights', '-w', type=str, default=None,
                        help='Optional weights for eng,act,invf as comma-separated floats (e.g. 0.6,0.2,0.2)')
    return parser.parse_args()

def main():
    args = parse_args()
    df = load_or_create_sample(args.input if os.path.exists(args.input) else None)

    # Aggregate per user
    agg = aggregate_user_stats(df)

    # Parse weights if provided
    weights = None
    if args.weights:
        try:
            a,b,c = [float(x) for x in args.weights.split(',')]
            s = a+b+c
            if s == 0:
                raise ValueError("Weights sum to zero")
            weights = {'eng': a/s, 'act': b/s, 'invf': c/s}
        except Exception as e:
            print("[WARN] Could not parse weights argument; using defaults. Error:", e)
            weights = None

    # Compute MIS
    result = compute_features_and_mis(agg, weights=weights)

    # Save and print
    save_and_print_results(result, args.output, topk=args.topk)

if __name__ == "__main__":
    main()
