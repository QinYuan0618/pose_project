from visualize_report import generate_report_charts

all_results = [
    {'video': '1.mp4', 'total_detections': 14567, 
     'counts': {'Standard': 6078, 'Slouching': 5241, 'Severe Forward/Lying': 703, 'Leaning': 2545}},
    {'video': '2.mp4', 'total_detections': 12067, 
     'counts': {'Standard': 4372, 'Slouching': 3719, 'Severe Forward/Lying': 2050, 'Leaning': 1926}},
    {'video': '3.mp4', 'total_detections': 14171, 
     'counts': {'Standard': 7235, 'Slouching': 3632, 'Severe Forward/Lying': 1396, 'Leaning': 1908}},
    {'video': '4.mp4', 'total_detections': 13469, 
     'counts': {'Standard': 2204, 'Slouching': 4785, 'Severe Forward/Lying': 5025, 'Leaning': 1455}}
]

generate_report_charts(all_results)