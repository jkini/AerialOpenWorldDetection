import json
import time
import os
from tabulate import tabulate
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# val_updated.json ----> ground-truth annotations file 
# Use this, instead of previous val.json
# Area of bboxes updated from previous zeros 
# Essential to bifurcate objects as small, medium, large 

# pred.json ----> prediction file 
# Sample data in pred.json 
# [{'image_id': 2356,
#   'category_id': 18,
#   'bbox': [1880, 2529, 17, 15],
#   'score': 1.0},
#  {'image_id': 2356,
#   'category_id': 73,
#   'bbox': [1773, 2581, 22, 34],
#   'score': 1.0}]

# eval_results.json ----> Output file (Summary)
# eval_detailed.txt ----> Output file (Detailed, per-category-wise)

def compute_metrics(model_id, gt_file, output_dir, pred_file, eval_results_file, eval_detailed_file):
    pred_file_path = os.path.join(output_dir, pred_file)
    eval_results_file_path = os.path.join(output_dir, eval_results_file)
    eval_detailed_file_path = os.path.join(output_dir, eval_detailed_file)
    
    print('##############################################')
    print('### COCO format evalution (All categories) ###')
    print('##############################################')
    
    # Start the timer
    start_time = time.time()
    
    # Load your custom COCO dataset
    xview_gt = COCO(gt_file)
    
    # Load your model's predictions
    with open(pred_file_path, 'r') as f:
        pred_data = json.load(f) 
    xview_dt = xview_gt.loadRes(pred_data)
    
    # Initialize COCO evaluation
    xview_eval = COCOeval(xview_gt, xview_dt, iouType='bbox')
    
    # Define custom area ranges
    small_area_threshold = 10 ** 2
    large_area_threshold = 100 ** 2
    small_area = [0, small_area_threshold]         # Small area is less than 10x10 pixels
    medium_area = [small_area_threshold, large_area_threshold]  # Medium area is 10x10 to 100x100 pixels
    large_area = [large_area_threshold, 1e5 ** 2]  # Large area is anything 100x100 pixels and larger
    
    # Modify areaRng and areaRngLbl attributes
    xview_eval.params.areaRng = [
        small_area,   # small
        medium_area,  # medium
        large_area,   # large
        [0, 1e5 ** 2] # all
    ]
    
    xview_eval.params.areaRngLbl = ['small', 'medium', 'large', 'all']
    
    # Modify maxDets for evaluation to handle thousands of detections per image
    # xview_eval.params.maxDets = [100, 1000, 3600]  # Adjusted to reflect the high number of detections
    
    # Run the evaluation
    xview_eval.evaluate()
    xview_eval.accumulate()
    xview_eval.summarize()
    
    # Optionally, save the results
    results = xview_eval.stats
    average_ap_small, average_ap_medium, average_ap_large = results[3:6]
    
    # Label the statistics for clarity
    labels = [
        "AP@[IoU=0.50:0.95]",
        "AP@[IoU=0.50]",
        "AP@[IoU=0.75]",
        "AP@[IoU=0.50:0.95 | area=small]",
        "AP@[IoU=0.50:0.95 | area=medium]",
        "AP@[IoU=0.50:0.95 | area=large]",
        "AR@[maxDets=1]",
        "AR@[maxDets=10]",
        "AR@[maxDets=100]",
        "AR@[maxDets=100 | area=small]",
        "AR@[maxDets=100 | area=medium]",
        "AR@[maxDets=100 | area=large]"
    ]
    
    # Create a dictionary to store labeled results
    results_dict = {label: stat for label, stat in zip(labels, results)}
    
    # Write the results to a JSON file
    with open(eval_results_file_path, 'w') as f:
        json.dump(results_dict, f, indent=4)
    
    print(f"Results successfully saved to {eval_results_file_path}")
    
    # End the timer
    end_time = time.time()
    
    # Calculate total time taken
    total_time = end_time - start_time
    print(f"Total processing time [Results successfully saved to {eval_results_file}]: {total_time:.2f} seconds")
    
    print('#####################################################')
    print('### Detailed COCO format evalution (Per-category) ###')
    print('#####################################################')
    
    # Start the timer
    start_time = time.time()
    
    # Initialize total counters for size categories
    total_small_obj = 0
    total_medium_obj = 0
    total_large_obj = 0
    
    # Prepare for category data and evaluations
    category_data = {}
    cat_ids = xview_gt.getCatIds()
    categories = xview_gt.loadCats(cat_ids)
    category_sizes = {cat['name']: {'small': 0, 'medium': 0, 'large': 0} for cat in categories}
    
    # Calculate size distributions
    for ann in xview_gt.anns.values():
        cat_name = xview_gt.loadCats([ann['category_id']])[0]['name']
        bbox_area = ann['bbox'][2] * ann['bbox'][3]  # width * height
        if bbox_area < small_area_threshold:
            category_sizes[cat_name]['small'] += 1
            total_small_obj += 1  # Update overall small counter
        elif bbox_area < large_area_threshold:
            category_sizes[cat_name]['medium'] += 1
            total_medium_obj += 1  # Update overall medium counter
        else:
            category_sizes[cat_name]['large'] += 1
            total_large_obj += 1  # Update overall large counter
    
    # Evaluate AP and AR for each category
    for cat_id, cat_name in zip(cat_ids, category_sizes):
        xview_eval.params.catIds = [cat_id]
        xview_eval.evaluate()
        xview_eval.accumulate()
        xview_eval.summarize()
        
        ap = xview_eval.stats[3:6]  # AP Small, Medium, Large
        overall_ap = xview_eval.stats[0]
        overall_ar = xview_eval.stats[8]
        
        # Calculate percentages
        sizes = category_sizes[cat_name]
        total = sum(sizes.values())
        percentages = {size: (count / total * 100) if total > 0 else 0 for size, count in sizes.items()}
        
        # Print percentages for debugging
        # print(f"Percentages for {cat_name}: {percentages}")
        
        category_data[cat_name] = {
            "GT Small (%)": f"{percentages['small']:.2f}%",
            "GT Medium (%)": f"{percentages['medium']:.2f}%",
            "GT Large (%)": f"{percentages['large']:.2f}%",
            "AP Small": ap[0],
            "AP Medium": ap[1],
            "AP Large": ap[2],
            "Overall AP": overall_ap,
            "Overall AR": overall_ar
        }
    
    # Calculate totals if needed
    # Calculate total number of annotations processed
    total_annotations = total_small_obj + total_medium_obj + total_large_obj
    overall_small_percentage = (total_small_obj / total_annotations * 100) if total_annotations > 0 else 0
    overall_medium_percentage = (total_medium_obj / total_annotations * 100) if total_annotations > 0 else 0
    overall_large_percentage = (total_large_obj / total_annotations * 100) if total_annotations > 0 else 0
    
    # Used from all-categories run (average_ap_small, average_ap_medium, average_ap_large)
    
    # Computed total AP and AR are averages of the values across all categories
    total_ap = sum(d['Overall AP'] for d in category_data.values()) / len(category_data)
    total_ar = sum(d['Overall AR'] for d in category_data.values()) / len(category_data)
    category_data['Total'] = {
        "GT Small (%)": overall_small_percentage,
        "GT Medium (%)": overall_medium_percentage,
        "GT Large (%)": overall_large_percentage,
        "AP Small": average_ap_small,
        "AP Medium": average_ap_medium,
        "AP Large": average_ap_large,
        "Overall AP": total_ap,
        "Overall AR": total_ar
    }
    
    # Format and print table
    headers = ["Category", "GT Small (%)", "GT Medium (%)", "GT Large (%)", "AP Small", "AP Medium", "AP Large", "Overall AP", "Overall AR"]
    table = [ [cat_name] + list(values.values()) for cat_name, values in category_data.items()]
    # print(tabulate(table, headers=headers, tablefmt='grid'))
    
    # Assuming 'table' is already created and contains all necessary data
    headers = ["Category", "GT Small (%)", "GT Medium (%)", "GT Large (%)", "AP Small", "AP Medium", "AP Large", "Overall AP", "Overall AR"]
    table_content = [ [cat_name] + list(values.values()) for cat_name, values in category_data.items()]
    formatted_table = tabulate(table_content, headers=headers, tablefmt='grid')
    
    # Write the formatted table to a text file
    with open(eval_detailed_file_path, 'w') as f:
        f.write(formatted_table)
    
    print(f"Table successfully saved to {eval_detailed_file_path}")
    
    # End the timer
    end_time = time.time()
    
    # Calculate total time taken
    total_time = end_time - start_time
    print(f"Total processing time [Results successfully saved to {eval_detailed_file}]: {total_time:.2f} seconds")
    
if __name__ == '__main__':
    model_id = 2
    gt_file = '/home/jy435956/Projects/OWLV2/datasets/val_updated.json'
    output_dir = f'/home/jy435956/Projects/OWLV2/outputs/{model_id}'
    pred_file = 'pred.json'
    eval_results_file = 'eval_results.json'
    eval_detailed_file = 'eval_detailed.txt'

    compute_metrics(model_id, gt_file, output_dir, pred_file, eval_results_file, eval_detailed_file)