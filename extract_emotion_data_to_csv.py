import os
import csv
import re
from pathlib import Path
import pandas as pd
from collections import defaultdict
# this is to extrat the csv from 
def parse_txt_file(txt_file_path):
    """Parse a single .txt file to extract emotion data"""
    data = {}
    
    try:
        with open(txt_file_path, 'r') as f:
            content = f.read()
            
        # Extract image name
        image_match = re.search(r'Image: (.+)', content)
        if image_match:
            data['image_name'] = image_match.group(1)
        
        # Extract expression
        expr_match = re.search(r'Expression: (.+)', content)
        if expr_match:
            data['expression'] = expr_match.group(1)
        
        # Extract expression confidence
        conf_match = re.search(r'Expression_Confidence: ([\d.-]+)', content)
        if conf_match:
            data['expression_confidence'] = float(conf_match.group(1))
        
        # Extract valence
        val_match = re.search(r'Valence: ([\d.-]+)', content)
        if val_match:
            data['valence'] = float(val_match.group(1))
        
        # Extract arousal
        ar_match = re.search(r'Arousal: ([\d.-]+)', content)
        if ar_match:
            data['arousal'] = float(ar_match.group(1))
        
        # Extract all expression probabilities
        prob_section = re.search(r'All Expression Probabilities:\n(.*)', content, re.DOTALL)
        if prob_section:
            prob_lines = prob_section.group(1).strip().split('\n')
            for line in prob_lines:
                if ':' in line:
                    emotion, prob = line.split(':', 1)
                    emotion = emotion.strip()
                    prob = float(prob.strip())
                    data[f'prob_{emotion.lower()}'] = prob
        
        return data
    
    except Exception as e:
        print(f"Error parsing {txt_file_path}: {e}")
        return None

def extract_all_emotion_data(base_path):
    """Extract emotion data from all .txt files in the directory structure"""
    base_path = Path(base_path)
    grouped_data = defaultdict(list)
    
    # Walk through all subdirectories
    for txt_file in base_path.rglob("*.txt"):
        # Skip the summary CSV files
        if txt_file.name == "emotion_results_summary.csv":
            continue
            
        # Get the relative path from base_path
        rel_path = txt_file.relative_to(base_path)
        
        # Extract directory structure (direction/emotion/level)
        if len(rel_path.parts) >= 4:  # direction/emotion/level/filename.txt
            direction = rel_path.parts[0]
            emotion = rel_path.parts[1] 
            level = rel_path.parts[2]
            group_key = f"{direction}/{emotion}/{level}"
        else:
            group_key = "ungrouped"
        
        # Parse the txt file
        data = parse_txt_file(txt_file)
        if data:
            data['file_path'] = str(txt_file)
            data['direction'] = direction if len(rel_path.parts) >= 4 else "unknown"
            data['emotion_category'] = emotion if len(rel_path.parts) >= 4 else "unknown"
            data['level'] = level if len(rel_path.parts) >= 4 else "unknown"
            grouped_data[group_key].append(data)
    
    return grouped_data

def save_grouped_csv_files(grouped_data, output_dir):
    """Save grouped data to separate CSV files"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    saved_files = []
    
    for group_key, data_list in grouped_data.items():
        if not data_list:
            continue
            
        # Create filename from group key
        filename = group_key.replace('/', '_') + '_emotion_data.csv'
        csv_path = output_dir / filename
        
        # Get all possible columns from all records
        all_columns = set()
        for data in data_list:
            all_columns.update(data.keys())
        
        # Sort columns for consistency
        columns = ['image_name', 'direction', 'emotion_category', 'level', 
                  'expression', 'expression_confidence', 'valence', 'arousal']
        
        # Add probability columns
        prob_columns = sorted([col for col in all_columns if col.startswith('prob_')])
        columns.extend(prob_columns)
        
        # Add any remaining columns
        remaining = sorted([col for col in all_columns if col not in columns])
        columns.extend(remaining)
        
        # Write CSV file
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            
            for data in data_list:
                # Fill missing columns with None/empty values
                row = {col: data.get(col, '') for col in columns}
                writer.writerow(row)
        
        saved_files.append(csv_path)
        print(f"✅ Saved {len(data_list)} records to {csv_path}")
    
    return saved_files

def create_master_csv(grouped_data, output_dir):
    """Create a single master CSV with all data"""
    output_dir = Path(output_dir)
    master_csv = output_dir / "master_emotion_data.csv"
    
    # Combine all data
    all_data = []
    for group_key, data_list in grouped_data.items():
        all_data.extend(data_list)
    
    if not all_data:
        print("No data to save to master CSV")
        return None
    
    # Get all possible columns
    all_columns = set()
    for data in all_data:
        all_columns.update(data.keys())
    
    # Sort columns for consistency
    columns = ['image_name', 'direction', 'emotion_category', 'level', 
              'expression', 'expression_confidence', 'valence', 'arousal']
    
    # Add probability columns
    prob_columns = sorted([col for col in all_columns if col.startswith('prob_')])
    columns.extend(prob_columns)
    
    # Add any remaining columns
    remaining = sorted([col for col in all_columns if col not in columns])
    columns.extend(remaining)
    
    # Write master CSV
    with open(master_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        
        for data in all_data:
            row = {col: data.get(col, '') for col in columns}
            writer.writerow(row)
    
    print(f"✅ Saved {len(all_data)} total records to {master_csv}")
    return master_csv

def generate_summary_stats(grouped_data, output_dir):
    """Generate summary statistics"""
    output_dir = Path(output_dir)
    summary_file = output_dir / "emotion_summary_stats.txt"
    
    with open(summary_file, 'w') as f:
        f.write("EMOTION DATA EXTRACTION SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        
        total_files = sum(len(data_list) for data_list in grouped_data.values())
        f.write(f"Total files processed: {total_files}\n")
        f.write(f"Total groups: {len(grouped_data)}\n\n")
        
        f.write("Files per group:\n")
        for group_key, data_list in sorted(grouped_data.items()):
            f.write(f"  {group_key}: {len(data_list)} files\n")
        
        # Expression distribution
        f.write("\nExpression distribution across all files:\n")
        expr_counts = defaultdict(int)
        for data_list in grouped_data.values():
            for data in data_list:
                if 'expression' in data:
                    expr_counts[data['expression']] += 1
        
        for expr, count in sorted(expr_counts.items()):
            f.write(f"  {expr}: {count}\n")
    
    print(f"✅ Summary statistics saved to {summary_file}")

def main():
    # Configuration
    result_output_path = "result_output"
    csv_output_dir = "emotion_csv_extracted"
    
    print("🔍 Extracting emotion data from .txt files...")
    
    # Extract all emotion data
    grouped_data = extract_all_emotion_data(result_output_path)
    
    if not grouped_data:
        print(f"❌ No .txt files found in {result_output_path}")
        return
    
    print(f"✅ Found {len(grouped_data)} groups with emotion data")
    
    # Save grouped CSV files
    print("\n📊 Saving grouped CSV files...")
    saved_files = save_grouped_csv_files(grouped_data, csv_output_dir)
    
    # Create master CSV
    print("\n📋 Creating master CSV file...")
    master_csv = create_master_csv(grouped_data, csv_output_dir)
    
    # Generate summary statistics
    print("\n📈 Generating summary statistics...")
    generate_summary_stats(grouped_data, csv_output_dir)
    
    print(f"\n{'='*60}")
    print("EXTRACTION COMPLETE")
    print(f"{'='*60}")
    print(f"📁 Output directory: {csv_output_dir}")
    print(f"📊 Individual CSV files: {len(saved_files)}")
    print(f"📋 Master CSV: {'Yes' if master_csv else 'No'}")
    
    # Show first few groups as examples
    print(f"\n📂 Example groups found:")
    for i, (group_key, data_list) in enumerate(sorted(grouped_data.items())[:5]):
        print(f"  {group_key}: {len(data_list)} files")
    if len(grouped_data) > 5:
        print(f"  ... and {len(grouped_data) - 5} more groups")

if __name__ == "__main__":
    main()