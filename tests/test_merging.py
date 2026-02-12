import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.modules.markdown_merger import MarkdownMerger

def test_smart_merging():
    print("Testing Smart Merging...")
    
    # Create dummy files
    file1 = "chunk_0.md"
    content1 = """# Video Title

## Table of Contents
- [Introduction](#introduction)
- [Section 1](#section-1)

## Introduction
Intro text.

## Section 1
Content 1.
"""
    with open(file1, "w") as f: f.write(content1)
    
    file2 = "chunk_1.md"
    content2 = """# Video Title (Part 2)

## Table of Contents
- [Section 2](#section-2)
- [Summary](#summary)

## Section 2
Content 2 with timestamp [05:00].

## Summary
Summary text.
"""
    with open(file2, "w") as f: f.write(content2)
    
    merger = MarkdownMerger()
    merged = merger.merge_markdowns([file1, file2], chunk_duration_minutes=10)
    
    print("\n--- Merged Content ---\n")
    print(merged)
    print("\n----------------------\n")
    
    # Verification
    if "## Table of Contents" in merged:
        print("✅ TOC present")
        if "- [Section 1](#section-1)" in merged and "- [Section 2](#section-2)" in merged:
             print("✅ Unified TOC contains both sections")
    
    if "# Video Title (Part 2)" not in merged:
        print("✅ Chunk 2 Title stripped")
        
    if merged.count("## Table of Contents") == 1:
        print("✅ Only one TOC present")
        
    if "Summary text" not in merged: # We strip Summary section from chunks > 0
        print("✅ Chunk 2 Summary stripped")
        
    if "[05:00]" not in merged and "[15:00]" in merged: # 10 min offset = 600s
        print("✅ Timestamp adjusted (05:00 -> 15:00)")
        
    # Cleanup
    os.remove(file1)
    os.remove(file2)

if __name__ == "__main__":
    test_smart_merging()
