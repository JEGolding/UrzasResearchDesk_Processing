# process_data.py
import pandas as pd
import numpy as np
import scipy.sparse
from sklearn.feature_extraction.text import CountVectorizer
import json
from pathlib import Path
from datetime import datetime, timedelta
# import umap
# import hdbscan
from vectorizers.transformers import InformationWeightTransformer
from tqdm import tqdm

def fuzzy_join(df1, df2):
    """
    Join two dataframes on 'Player' column, handling duplicate names by matching based on closest rank.
    This handles for when there are duplicate player names in an event.
    
    Parameters:
    df1, df2: Pandas DataFrames with 'Player' and 'Rank' columns
    
    Returns:
    Pandas DataFrame with joined results
    """
    # Step 1: Do a standard join on names first
    # This will work for all unique names
    standard_join = pd.merge(df1, df2, on='Player', how='inner', suffixes=('','_standings'))
    
    # Step 2: Find duplicate names from both dataframes
    duplicate_names_df1 = df1['Player'].value_counts()[df1['Player'].value_counts() > 1].index.tolist()
    duplicate_names_df2 = df2['Player'].value_counts()[df2['Player'].value_counts() > 1].index.tolist()
    duplicate_names = list(set(duplicate_names_df1 + duplicate_names_df2))
    
    # Step 3: Remove duplicate named rows from the standard join
    clean_join = standard_join[~standard_join['Player'].isin(duplicate_names)]
    
    # Step 4: Handle duplicates separately
    fuzzy_results = []
    for dup_name in duplicate_names:
        # Get all rows with this name from both dataframes
        dup_df1 = df1[df1['Player'] == dup_name].copy()
        dup_df2 = df2[df2['Player'] == dup_name].copy()
        
        # If we have duplicates in both dataframes, we need to do fuzzy matching
        if len(dup_df1) > 0 and len(dup_df2) > 0:
            # Create a distance matrix between all rank combinations
            distances = np.zeros((len(dup_df1), len(dup_df2)))
            
            for i, row1 in enumerate(dup_df1.itertuples()):
                for j, row2 in enumerate(dup_df2.itertuples()):
                    distances[i, j] = abs(row1.Rank - row2.Rank)
            
            # Match rows greedily by minimum rank distance
            matched_pairs = []
            while len(matched_pairs) < min(len(dup_df1), len(dup_df2)):
                # Find the minimum distance
                min_idx = np.unravel_index(distances.argmin(), distances.shape)
                matched_pairs.append((min_idx[0], min_idx[1]))
                
                # Mark this pair as matched by setting distance to infinity
                distances[min_idx[0], :] = np.inf
                distances[:, min_idx[1]] = np.inf
            
            # Create joined rows based on matched pairs
            for df1_idx, df2_idx in matched_pairs:
                row_df1 = dup_df1.iloc[df1_idx]
                row_df2 = dup_df2.iloc[df2_idx]
                
                joined_row = pd.DataFrame({
                    'name': [row_df1['Player']],
                    'rank_df1': [row_df1['Rank']],
                    'rank_df2': [row_df2['Rank']]
                })
                
                fuzzy_results.append(joined_row)
    
    # Step 5: Combine standard join with fuzzy results
    if fuzzy_results:
        fuzzy_join = pd.concat(fuzzy_results, ignore_index=True)
        final_result = pd.concat([clean_join, fuzzy_join], ignore_index=True)
    else:
        final_result = clean_join
    
    return final_result

def get_tournament_files(base_path='../MTGODecklistCache/Tournaments', lookback_days=365, fmt='modern'):
    """
    Find all modern tournament files from the last lookback_days.
    
    Parameters:
    -----------
    base_path : str
        Path to tournament data directory
    lookback_days : int
        Number of days to look back
    fmt : str
        Tournament format
        
    Returns:
    --------
    list
        List of Path objects for matching tournament files
    """
    cutoff_date = datetime.now() - timedelta(days=lookback_days)
    
    # Get all possible year/month/day combinations from cutoff to now
    date_range = []
    current_date = cutoff_date
    while current_date <= datetime.now():
        date_range.append(current_date)
        current_date += timedelta(days=1)
    
    # Create patterns for each date
    patterns = [
        f"*/{date.year}/{date.month:02d}/{date.day:02d}/*{fmt}*.json"
        for date in date_range
    ]
    
    # Find all matching files
    matching_files = []
    base_path = Path(base_path)
    for pattern in patterns:
        matching_files.extend(base_path.glob(pattern))

    if not matching_files:
        raise ValueError('No valid file paths were found.')
    
    return matching_files

def process_mtg_data(lookback_days=365, fmt='Modern'):
    """Process MTG tournament data and save results for dashboard consumption."""

    print(f'Processing {fmt} tournament files')

    # Initialize empty DataFrame
    df = pd.DataFrame()
    
    # Process tournament files
    tournament_path = Path('../MTGODecklistCache/Tournaments/')
    # Add tqdm back here if needed.
    for path in get_tournament_files(tournament_path, lookback_days, fmt.lower()):
        try:
            with open(path) as f:
                data = json.load(f)
            
            deck_df = pd.DataFrame(data['Decks'])
            deck_df['Deck'] = data['Decks']
            deck_df['Tournament'] = path.name
            
            # Process standings
            standings_df = pd.DataFrame(data['Standings'])
            if standings_df.shape[0]:
                if deck_df.loc[0, 'Result'].endswith('Place'):
                    deck_df['Rank'] = deck_df['Result'].str[:-8].astype(int)
                else:
                    deck_df['Rank'] = range(deck_df.shape[0])
                deck_df = fuzzy_join(deck_df, standings_df)

                if deck_df['Wins'].sum() == deck_df['Losses'].sum():
                    # Everything is fine.
                    #
                    deck_df['Invalid_WR'] = False
                elif data['Rounds'] is not None:
                    # We need to build the win rates from the individual rounds.
                    #
                    round_df = pd.concat([pd.DataFrame(r['Matches']) for r in data['Rounds']], ignore_index=True)

                    # Some players we don't have deck lists for, so we shouldn't include them in the wr.
                    #
                    round_df = round_df[
                        round_df['Player1'].isin(deck_df['Player']) & round_df['Player2'].isin(deck_df['Player'])
                    ]
                    
                    for i in deck_df.index:
                        # In order, 
                        # Make sure our player won/lost,
                        # Make sure it wasn't a draw,
                        # Make sure it wasn't a bye.
                        deck_df.loc[i, 'Wins'] = (
                            (round_df['Player1'] == deck_df.loc[i, 'Player']) & \
                            round_df['Result'].str.startswith('2') & \
                            ~(round_df['Player2'] == ('-'))
                        ).sum(axis=None)
                        deck_df.loc[i, 'Losses'] = (
                            (round_df['Player2'] == deck_df.loc[i, 'Player']) & \
                            round_df['Result'].str.startswith('2')
                        ).sum(axis=None)
                    
                    if deck_df['Wins'].sum() == deck_df['Losses'].sum():
                        # Everything is fine.
                        #
                        deck_df['Invalid_WR'] = False
                    else:
                        deck_df['Invalid_WR'] = True
                        # print(deck_df['Player'].nunique(), deck_df.shape)
                        # print(path, deck_df['Wins'].sum(), deck_df['Losses'].sum())
                        print(f'Could not fix {path}')

                elif 'mtgo.com' in str(path):
                    # Draws can't happen, we can look at points.
                    # Sometimes wins aren't recorded.
                    # Could be the same for losses, but we can't do anything about that.
                    # We'll fix for wins and if things are still broken call it.
                    #
                    deck_df['Wins'] = deck_df['Points'] / 3

                    if deck_df['Wins'].sum() == deck_df['Losses'].sum():
                        # Everything is fine.
                        #
                        deck_df['Invalid_WR'] = False
                    else:
                        deck_df['Invalid_WR'] = True
                        # print(deck_df['Player'].nunique(), deck_df.shape)
                        # print(path, deck_df['Wins'].sum(), deck_df['Losses'].sum())
                        print(f'Could not fix mtgo event {path}')

                else:
                    deck_df['Invalid_WR'] = True
            else:
                deck_df['Invalid_WR'] = True

            
            # Set date from path if missing
            deck_df['Date'] = f'{path.parent.parent.parent.name}-{path.parent.parent.name}-{path.parent.name}'
            
            df = pd.concat([df, deck_df], ignore_index=True)
        except Exception as e:
            print(path)
            raise e
        
    if not df.shape[0]:
        raise ValueError('No data was found in the specified files.')
    
    # Convert dates and sort
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by='Date')

    df = df.dropna(subset=['Date','Deck'])

    print(f'deck data loaded, shape={df.shape}')
    print(f'Invalid win rates: shape=({df["Invalid_WR"].sum()})')

    # # Load card data
    # with open('../AtomicCards.json', 'r') as f:
    #     j = json.load(f)['data']
    # card_list = j.keys()

    # print(f'card data loaded, shape={len(card_list)}')
    
    # Vectorize decks
    def merge_analyzer(deck):
        """Convert deck dictionary into list of card strings."""
        output = []
        for card in deck['Mainboard']:
            # if card['CardName'] in card_list:
            #     if 'Land' not in j[card['CardName']][0]['type']:
            #         output += [card['CardName']] * card['Count']
            # else:
            output += [card['CardName']] * card['Count']
        for card in deck['Sideboard']:
            output += [card['CardName']+'_SB'] * card['Count']
        return output

    vectorizer = CountVectorizer(analyzer=merge_analyzer)
    X = vectorizer.fit_transform(df['Deck'])
    
    # Apply Information Weight Transform
    iwt = InformationWeightTransformer()
    X_iwt = iwt.fit_transform(X)

    print('Vectorized')
    
    # # Apply UMAP for dimensionality reduction
    # reducer = umap.UMAP(
    #     n_components=3,
    #     metric='cosine',
    #     # n_neighbors=15,
    #     # min_dist=0.1,
    #     random_state=42
    # )
    
    # X_umap = reducer.fit_transform(X_iwt)

    # print('UMAP complete')
    
    # # Perform clustering on UMAP embedding
    # clusterer = hdbscan.HDBSCAN(
    #     min_cluster_size=100,
    #     min_samples=5,
    #     cluster_selection_epsilon=0.5,
    #     metric='euclidean'  # Use euclidean for reduced dimensions
    # )
    
    # cluster_labels = clusterer.fit_predict(X_umap)

    # print(f'HDBSCAN complete, {clusterer.labels_.max()} clusters')
    
    # # Calculate cluster representatives
    # cluster_representatives = {}
    # for label in tqdm(range(clusterer.labels_.max()+1)):
    #     # Get decks in this cluster
    #     cluster_mask = cluster_labels == label
    #     cluster_vectors = X[cluster_mask]
        
    #     # Calculate mean card counts
    #     mean_counts = cluster_vectors.mean(axis=0).A1
    #     std_counts = cluster_vectors.toarray().std(axis=0)
        
    #     # Get top cards by mean/std ratio
    #     card_stats = list(zip(
    #         mean_counts,
    #         std_counts,
    #         vectorizer.get_feature_names_out()
    #     ))
        
    #     # Sort by mean/std ratio, handling divide by zero
    #     top_cards = sorted([
    #         (m, s, n) for m, s, n in card_stats
    #         if m > 0.1  # Only include cards that appear in at least 10% of decks
    #     ], key=lambda x: x[0]/(x[1] if x[1] > 0 else 0.1), reverse=True)[:10]
        
    #     cluster_representatives[label] = {
    #         'size': int(cluster_mask.sum()),
    #         'win_rate': float(df.loc[cluster_mask, 'Wins'].sum() / 
    #                         (df.loc[cluster_mask, 'Wins'].sum() + df.loc[cluster_mask, 'Losses'].sum())),
    #         'top_cards': [{'name': n, 'mean': float(m), 'std': float(s)} for m, s, n in top_cards]
    #     }

    # print('Clusters analysed')
    
    # Create output directory
    Path('processed_data').mkdir(exist_ok=True)

    # Generate and save metadata
    metadata = {
        'last_updated': datetime.utcnow().isoformat(),
        'num_decks': df.shape[0],
        'date_range': [df['Date'].min().isoformat(), df['Date'].max().isoformat()],
        # 'num_clusters': len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    }
    
    with open(f'processed_data/metadata.json', 'w') as f:
        json.dump(metadata, f)
    
    df['Date'] = df['Date'].astype(str)
    # Save processed data
    output_data = {
        'decks': df[['Player', 'Wins', 'Losses', 'Date', 'Tournament', 'Invalid_WR']].to_dict('records'),
        'clusters': [],#cluster_labels.tolist(),
        'cluster_info': [],#cluster_representatives,
        'feature_names': vectorizer.get_feature_names_out().tolist()
    }
    
    with open(f'processed_data/deck_data.json', 'w') as f:
        json.dump(output_data, f)
    
    # Save matrices
    scipy.sparse.save_npz(f'processed_data/card_vectors.npz', X)
    # np.save('processed_data/umap_embedding.npy', X_umap)
    
    # Save transformers data
    vectorizer_data = {
        'vocabulary': vectorizer.vocabulary_
    }
    with open(f'processed_data/vectorizer.json', 'w') as f:
        json.dump(vectorizer_data, f)
        
    # iwt_data = {
    #     'idf': iwt.idf_.tolist()
    # }
    # with open('processed_data/iwt.json', 'w') as f:
    #     json.dump(iwt_data, f)

    print('Data saved, done')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("format", help="Format to process", default='Modern')
    args = parser.parse_args()

    process_mtg_data(fmt=args.format)#lookback_days=30)
