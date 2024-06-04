# -*- encoding: utf-8 -*-
'''
@File    :   pre_processing_rj.py
@Author  :   Yang Liu (yang.liu3@halliburton.com) 
'''
# import lib below

import os
import sys
import numpy as np
import pandas as pd
from copy import deepcopy

def data_preprocessing(survey_inc, cont_survey, live_inc, start_depth, end_depth, start_idx, end_idx, dc_rjmcmc, df_TY, window_size, data_pts_thres):
    survey_window, cont_window, live_window = create_window_data(survey_inc, cont_survey, live_inc, start_depth, end_depth, start_idx, end_idx)
    selected_df = concatenate_window_data(live_window, survey_window, cont_window, dc_rjmcmc)
    segments = locate_changepoint_and_segment(selected_df)
    selected_segments = select_segment_per_mode(segments, data_pts_thres)
    if df_TY.empty:
        selected_segments_pre = []
    else:
        end_depth_pre = df_TY['depth'].iloc[-1]
        end_idx_pre = live_inc.index[abs(live_inc['depth']- end_depth_pre) == min(abs(live_inc['depth'] - end_depth_pre))]#live_inc.index[live_inc['depth'] == end_depth_pre]
        end_idx_pre = end_idx_pre[0] + 1
        start_idx_pre = end_idx_pre - window_size
        if start_idx_pre < 0:
            start_idx_pre = 0
        start_depth_pre = live_inc['depth'].iloc[start_idx_pre]
        survey_window_pre, cont_window_pre, live_window_pre = create_window_data(survey_inc, cont_survey, live_inc, start_depth_pre, end_depth_pre, start_idx_pre, end_idx_pre)
        selected_df_pre = concatenate_window_data(live_window_pre, survey_window_pre, cont_window_pre, dc_rjmcmc)
        segments_pre = locate_changepoint_and_segment(selected_df_pre)
        selected_segments_pre = select_segment_per_mode(segments_pre, data_pts_thres)
    for seg_prev in selected_segments_pre:
        selected_segments = [seg for seg in selected_segments if not seg.equals(seg_prev)]
        
    return selected_segments, live_window, survey_window, cont_window

def retrieve_from_dataset(dataset, contsurvey, incthres):
    dc_rjmcmc = dataset['dc']
    tf_rjmcmc = dataset['tf']
    survey_inc = dataset['survey'][dataset['survey']['inc'] >= incthres].reset_index(drop=True)
    live_inc = dataset['inc']['pcdcinc']
    live_inc = live_inc[(live_inc['val'] >= incthres)].reset_index(drop=True)
    cont_survey = pd.DataFrame({'depth': [], 'val': []})
    cont_survey['depth'] = contsurvey['depthinc']
    cont_survey['val'] = contsurvey['inc']
   
    return dc_rjmcmc, tf_rjmcmc, survey_inc, live_inc, cont_survey
 
def get_mode_noise(live_inc, dc_rjmcmc):
    live_inc_copy = deepcopy(live_inc)
    live_inc_copy = live_inc_copy.reset_index(drop=True)
    window_size = 3
    padding = (window_size - 1) // 2  # Half of the window size
    padded_data = pd.concat([live_inc['val'].head(padding), live_inc['val'], live_inc['val'].tail(padding)], ignore_index=True)
    smooth = np.convolve(padded_data, np.ones(window_size) / window_size, mode='valid')
    bias = smooth - live_inc['val'] # store for noise estimation in calibration
    bias = bias.reset_index(drop=True)
    live_inc_copy['bias'] = bias
   
    for i in range(len(live_inc_copy)):
        depth_rjmcmc = live_inc_copy['depth'].iloc[i]
        filtered_dc = dc_rjmcmc[dc_rjmcmc['depth'] < depth_rjmcmc]
       
        if not filtered_dc.empty:
            dc_depth = filtered_dc['depth'].max()
            mode_value = dc_rjmcmc.loc[dc_rjmcmc['depth'] == dc_depth, 'val'].values[0]
            live_inc_copy.at[i, 'mode'] = mode_value
   
    live_inc_slide = live_inc_copy[live_inc_copy['mode'] == 100]
    live_inc_rotate = live_inc_copy[live_inc_copy['mode'] == 0]
    sigma_slide = np.std(live_inc_slide['bias'], ddof=1)/2
    sigma_rotate= np.std(live_inc_rotate['bias'], ddof=1)/2
   
    return sigma_slide, sigma_rotate
 
def create_window_data(survey_inc, cont_survey, live_inc, start_depth, end_depth, start_idx, end_idx):
    live_window = pd.DataFrame({'depth': [], 'val': [], 'bias': [], 'mode': []})
    live_window['depth'] = live_inc['depth'].iloc[start_idx:end_idx]
    live_window['val'] = live_inc['val'].iloc[start_idx:end_idx]
    live_window['bias'] = live_inc['bias'].iloc[start_idx:end_idx]
    cont_window = cont_survey[(cont_survey['depth'] >= start_depth) & (cont_survey['depth'] <= end_depth)]
    survey_window = survey_inc[(survey_inc['depth'] >= start_depth) & (survey_inc['depth'] <= end_depth)]
   
    return survey_window, cont_window, live_window
 
def concatenate_window_data(live_window, survey_window, cont_window, dc_rjmcmc):
    live_selected = live_window[['depth', 'val']]
    survey_selected = survey_window[['depth', 'inc']]
    survey_selected = survey_selected.rename(columns={'inc': 'val'})
    selected_df = pd.concat([live_selected, survey_selected, cont_window], axis=0)
    selected_df = selected_df.sort_values(by='depth', ascending=True).reset_index(drop=True)
   
    for idx, row in selected_df.iterrows():
        depth_rjmcmc = row['depth']
        filtered_dc = dc_rjmcmc[dc_rjmcmc['depth'] < depth_rjmcmc]
       
        if not filtered_dc.empty:
            dc_depth = filtered_dc['depth'].max()
            mode_value = dc_rjmcmc.loc[dc_rjmcmc['depth'] == dc_depth, 'val'].values[0]
            selected_df.at[idx, 'mode'] = mode_value
   
    return selected_df
 
def locate_changepoint_and_segment(selected_df):
    segments = []
    change_points = np.where(np.diff(selected_df['mode']) != 0)[0] + 1
   
    for i in range(len(change_points)+1):
        start = change_points[i - 1] if i > 0 else 0
        end = change_points[i] if i < len(change_points) else len(selected_df)
        segment = selected_df[start:end].reset_index(drop=True)
        segments.append(segment)
   
    return segments
 
def select_segment_per_mode(segments, data_pts_threshold):
    selected_segments = []
    recent_segment = None
    prev_segment = None  
    for sgmt in reversed(segments):
        if len(sgmt) >= data_pts_threshold:
            recent_segment = sgmt
            if prev_segment is not None:
                if recent_segment['mode'].iloc[0] != prev_segment['mode'].iloc[0]:
                    selected_segments.append(recent_segment)
                    break
                elif recent_segment['mode'].iloc[0] == prev_segment['mode'].iloc[0]:
                    break
            selected_segments.append(recent_segment)
            prev_segment = recent_segment
   
    return selected_segments
 
def update_window_data(selected_segments, live_window, survey_window, cont_window):
    selected_df_filtered = pd.concat(selected_segments, ignore_index=True)
    selected_df_filtered = selected_df_filtered.sort_values(by='depth', ascending=True).reset_index(drop=True)
    selected_df_filtered['bias'] = [None]*len(selected_df_filtered)
   
    liveinc_rj = live_window.loc[live_window['depth'].isin(selected_df_filtered['depth']),['depth', 'val', 'bias']].reset_index(drop=True)
    # liveinc_rj = liveinc_rj.merge(selected_df_filtered[['depth', 'mode', 'tf']], on = 'depth', how='left')
    liveinc_rj = liveinc_rj.merge(selected_df_filtered[['depth', 'mode']], on = 'depth', how='left')
    survey_rj = survey_window.loc[survey_window['depth'].isin(selected_df_filtered['depth'])].reset_index(drop=True)
    cont_rj = cont_window.loc[cont_window['depth'].isin(selected_df_filtered['depth'])].reset_index(drop=True)
 
    return liveinc_rj, survey_rj, cont_rj, selected_df_filtered
 
def slide_segment_range(selected_segments, df_TY):
    slide_df_filtered = [df for df in selected_segments if df.at[0, 'mode'] == 100]
    if selected_segments and slide_df_filtered:
        slide_df_start_depth = slide_df_filtered[0]['depth'].iloc[0]
        slide_df_end_depth = slide_df_filtered[0]['depth'].iloc[-1]
    elif (not bool(slide_df_filtered)) and not df_TY.empty:
        slide_df_start_depth = df_TY['slide_start'].iloc[-1]
        slide_df_end_depth = df_TY['slide_end'].iloc[-1]  
    else:
        slide_df_start_depth = None
        slide_df_end_depth = None      
   
    return slide_df_start_depth, slide_df_end_depth
 
def update_toolyield_dataframe(toolyield, df_TY, slide_start, slide_end, priors):
    slide_flag = 0
    rotate_flag = 0
   
    if toolyield['slide'].iloc[0] is None:
        if df_TY.shape[0] != 0:
            if df_TY['slide_flag'].iloc[-1] == 0:
                toolyield['slide'].iloc[0] = priors['kact']
                print("NOT SUFFICIENT LIVE INC DATA FOR SLIDE MODE, RESORT TO PRIOR =============")
           
            else:
                slide_flag = 1
                toolyield['slide'].iloc[0] = df_TY['slide'].iloc[-1]
                print("SLIDE TOOLYIELD IS NOT UPDATED, RESORT TO PREVIOUS RESULT ================")
        else:
            toolyield['slide'].iloc[0] = priors['kact']
            print("NOT SUFFICIENT LIVE INC DATA FOR SLIDE MODE, RESORT TO PRIOR =============")
           
    else:
        if toolyield['slide'].iloc[0] < 0.0 or toolyield['slide'].iloc[0] >= 30:
            toolyield['slide'].iloc[0] = priors['kact']
            print("SLIDE TOOLYIELD IS OUT OF REASONABLE RANGE, RESORT TO PRIOR ==============")
        else:
            slide_flag = 1
       
    if toolyield['rotate'].iloc[0] is None:
        if df_TY.shape[0] != 0:
            if df_TY['rotate_flag'].iloc[-1] == 0:
                toolyield['rotate'].iloc[0] = priors['kbiasi']
                print("NOT SUFFICIENT LIVE INC DATA FOR ROTATE MODE, RESORT TO PRIOR ============")
            else:
                rotate_flag = 1
                toolyield['rotate'].iloc[0] = df_TY['rotate'].iloc[-1]
                print("ROTATE TOOLYIELD IS NOT UPDATED, RESORT TO PREVIOUS RESULT ===============")            
        else:
            toolyield['rotate'].iloc[0] = priors['kbiasi']
            print("NOT SUFFICIENT LIVE INC DATA FOR ROTATE MODE, RESORT TO PRIOR ============")
    else:
        if (toolyield['rotate'].iloc[0] > 6.0) or (toolyield['rotate'].iloc[0] < -6.0):
            toolyield['rotate'].iloc[0] = priors['kbiasi']
            print("ROTATE TOOLYIELD IS OUT OF REASONABLE RANGE, RESORT TO PRIOR =============")
        else:
            rotate_flag = 1
       
    new_row = pd.DataFrame({'depth': [toolyield.iloc[0]['depth']],  # Ensure depth is in a list
                            'slide': [toolyield.iloc[0]['slide']],
                            'rotate': [toolyield.iloc[0]['rotate']],
                            'slide_start': [slide_start],
                            'slide_end': [slide_end],
                            'slide_flag': [slide_flag],
                            'rotate_flag': [rotate_flag]})

    # Append the new row to df_TY
    df_TY = pd.concat([df_TY, new_row], ignore_index=True)

       
    # df_TY = df_TY.append({'depth': toolyield.iloc[0]['depth'],
    #                     'slide': toolyield.iloc[0]['slide'],
    #                     'rotate': toolyield.iloc[0]['rotate'],
    #                     'slide_start': slide_start,
    #                     'slide_end': slide_end,
    #                     'slide_flag': slide_flag,
    #                     'rotate_flag': rotate_flag}, ignore_index=True)
    return df_TY