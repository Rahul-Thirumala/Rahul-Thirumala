"""
Sub-module: Record Linkage (Sugamya/CodeS─ümarthya/GOS/DASH/DPS)

...

* Integrates with dps_1_0.py after validation stage, i.e.,
    stage 4 of GOS/DASH/DPS.

Needs to solve Record Linkage Problems between:
    * Front Desk Records from Notion
    
    * ICICI Bank Settlements
    
    * PayTM settlements & payments
    
    * Booking.com reservations
    
    * InGo-MMT Payments & Bookings

Ongoing work
------------
Solving record linkage problem between:
    * Front Desk Records from Notion
    * PayTM payments

Pending Work
------------
All the rest of record linkage

Future Work
-----------

Module Dependencies
-------------------
* strings matching - (Nidhi Shastry)

Author(s)
---------
    *Rahul Thirumala (thirumala.rahul@gmail.com)
        VVCE, Vijayangara 2nd Stage

    *Nidhi Shastry (nidhishastry08@gmail.com)
        VVCE, Vijayangara 2nd Stage


Project Ownership
-----------------
Gaurav S Hegde (grv.hegde@gmail.com)
CodeS─ümarthya, SAfE

References
----------

Date Last Modified
------------------
Significantly modified on:
April 20, 2024
"""

# third party packages
import pandas as pd
import numpy as np

# dev modules
import dash_loader_modified3 as dfs
from dps_1_0_rl_sm import match_strings

# standard library
from datetime import datetime
import re
import os
import pdb

# Set the display options for pandas
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


# STEP 0: PREPARATION STEP

# Pre-processing Steps for fd
fd_original = dfs.load_front_desk('./datasets/front_desk_testcase_02.csv')[
    ['Record Index FD', 'Name', 'Phone', 'UPI Details']
]
ptmt = dfs.load_paytm_payments('./datasets/paytm_payments_testcase_02.csv')[
    ['Record Index PTMT', 'Transaction_Date', 'Amount', 'Customer_VPA']
]
fd_original['UPI Details'] = fd_original['UPI Details'].apply(
    lambda x: list(zip(*x)))

fd = fd_original.explode('UPI Details', ignore_index=True)
fd[['UPI_Date', 'UPI_Name', 'UPI_Amt']] = pd.DataFrame(
    fd['UPI Details'].tolist(),
    columns=['UPI_Date', 'UPI_Name', 'UPI_Amt'],
    index=fd.index
)
fd['UPI_Amt'] = fd['UPI_Amt'].astype(np.float64)
fd['UPI_Date'] = fd['UPI_Date'].apply(lambda x: pd.to_datetime(
    datetime.strptime(x, "%d%m%Y").strftime("%Y-%m-%d")
))
fd.drop('UPI Details', axis=1, inplace=True)

# Pre-processing steps for ptmt
ptmt['Transaction_Date_rl'] = pd.to_datetime(
    pd.to_datetime(ptmt['Transaction_Date']).dt.strftime('%Y-%m-%d')
)
ptmt['name_ptmt'] = ptmt['Customer_VPA'].apply(
    lambda vpa: re.sub(r'[^a-zA-Z]', '', vpa.strip("'").split('@')[0])
    if not re.match(r'^\'?(\d{10,12})@', vpa) else np.nan
)
ptmt['phone_ptmt'] = ptmt['Customer_VPA'].apply(
    lambda vpa: re.match(r'^\'?(\d{10,12})@', vpa).group(1)
    if re.match(r'^\'?(\d{10,12})@', vpa) else np.nan
)

ptmt.drop('Customer_VPA', axis=1, inplace=True)


# Temporary variables:
# Intermediate store of indicies before partitioning fd
# At every step, we may need to check only within these indices
case_indices = {
    'ABCD': {
        'fd': [], 'ptmt': []
    },
    'EFGH': {
        'fd': [], 'ptmt': []
    },
    'AB': {
        'fd': [], 'ptmt': []
    },
    'B': {
        'fd': [], 'ptmt': []
    },
    'CD': {
        'fd': [], 'ptmt': []
    },
    'EF': {
        'fd': [], 'ptmt': []
    },
    'GH': {
        'fd': [], 'ptmt': []
    },
    'iter2': {},  # for later
    'C': {
        'fd': [], 'ptmt': []
    },
    'D': {
        'fd': [], 'ptmt': []
    },
}

# hold final matching records
matches = {'left_index': [], 'right_index': [], 'label': []}

case_indices["EFGH"]["fd"].extend(
    fd.index[~(fd["UPI_Date"].isin(ptmt['Transaction_Date_rl']))].to_list()
)
case_indices["EFGH"]["ptmt"].extend(
    ptmt.index[~(ptmt["Transaction_Date_rl"].isin(fd["UPI_Date"]))].to_list()
)

temp_grouped_fd = fd[fd["UPI_Date"]
                      .isin(ptmt['Transaction_Date_rl'])]\
                      .groupby('UPI_Date')
temp_grouped_ptmt = ptmt[ptmt["Transaction_Date_rl"]
                         .isin(fd["UPI_Date"])]\
                         .groupby('Transaction_Date_rl')

for upi_date, group_fd in temp_grouped_fd:
    group_ptmt = temp_grouped_ptmt.get_group(upi_date)
    
    if len(group_fd) == len(group_ptmt):
        case_indices["ABCD"]["fd"].extend(group_fd.index)
        case_indices["ABCD"]["ptmt"].extend(group_ptmt.index)
    else:
        case_indices["EFGH"]["fd"].extend(group_fd.index)
        case_indices["EFGH"]["ptmt"].extend(group_ptmt.index)

fd_group = fd.loc[case_indices["ABCD"]["fd"]].groupby('UPI_Date')
ptmt_group =\
    ptmt.loc[case_indices["ABCD"]["ptmt"]].groupby('Transaction_Date_rl')

for key in fd_group.groups:
    # lists of records that match on amounts, a pair of list - one each from fd
    # and ptmt keys - corresponds to a match on date. Records corresponding to
    # indices will also have same amounts but will be pending matching
    case_indices["AB"]["fd"].append(
        fd_group.get_group(key).index[
            fd_group.get_group(key)['UPI_Amt'].isin(
                ptmt_group.get_group(key)['Amount']
            )
        ].to_list()
    )
    case_indices["AB"]["ptmt"].append(
        ptmt_group.get_group(key).index[
            ptmt_group.get_group(key)['Amount'].isin(
                fd_group.get_group(key)['UPI_Amt']
            )
        ].to_list()
    )
    # same as above but those that don't match on the amounts
    case_indices["CD"]["fd"].extend(
        fd_group.get_group(key).index[
            ~(fd_group.get_group(key)['UPI_Amt'].isin(
                ptmt_group.get_group(key)['Amount']
            ))
        ].to_list()
    )
    case_indices["CD"]["ptmt"].extend(
        ptmt_group.get_group(key).index[
            ~(ptmt_group.get_group(key)['Amount'].isin(
                fd_group.get_group(key)['UPI_Amt']
            ))
        ].to_list()
    )
pdb.set_trace()
for i in range(len(case_indices["AB"]["fd"])):
    idx_left = case_indices["AB"]["fd"][i]
    idx_right = case_indices["AB"]["ptmt"][i]

    # amount-wise name matching
    fd_group = fd.loc[idx_left].groupby("UPI_Amt")
    ptmt_group = ptmt.loc[idx_right].groupby("Amount")
    for key in fd_group.groups:
        if ptmt_group.get_group(key)['name_ptmt'].notnull().any():
            temp = match_strings(
                fd_group.get_group(key)['UPI_Name'],
                ptmt_group.get_group(key)['name_ptmt'][
                    ptmt_group.get_group(key)['name_ptmt'].notnull()
                ],
                match_type='name',
                threshold=0.37,
            )
        if ptmt_group.get_group(key)['phone_ptmt'].notnull().any():
            temp = match_strings(
                fd_group.get_group(key)['Phone'],
                ptmt_group.get_group(key)['phone_ptmt'][
                    ptmt_group.get_group(key)['phone_ptmt'].notnull()
                ],
                match_type='phone',  # Use match_type 'phone'
                threshold=0.9,       # Use threshold 1 for 'phone' matching
            )
        # case 'A'
        matches['left_index'].extend(temp[0])
        matches['right_index'].extend(temp[1])
        matches['label'].extend(['perfect' for i in range(len(temp[0]))])

        # case 'B' (potential iter2)
        # currently match with the closest names/phone numbers
        temp1 = [[], []]
        if len(fd_group.get_group(key)) > len(temp[0]) and \
           len(ptmt_group.get_group(key)) > len(temp[1]):
            if ptmt_group.get_group(key)['name_ptmt'].notnull().any():
                temp1 = match_strings(
                    fd_group.get_group(key).loc[
                        ~fd_group.get_group(key).index.isin(temp[0])
                    ]['UPI_Name'],
                    ptmt_group.get_group(key).loc[
                        ~ptmt_group.get_group(key).index.isin(temp[1])
                    ]['name_ptmt'][ptmt_group.get_group(
                        key)['name_ptmt'].notnull()],
                    match_type='name',
                    threshold=0,
                )
            if ptmt_group.get_group(key)['phone_ptmt'].notnull().any():
                temp1 = match_strings(
                    fd_group.get_group(key).loc[
                        ~fd_group.get_group(key).index.isin(temp[0])
                    ]['Phone'],
                    ptmt_group.get_group(key).loc[
                        ~ptmt_group.get_group(key).index.isin(temp[1])
                    ]['phone_ptmt'][ptmt_group.get_group(
                        key)['phone_ptmt'].notnull()],
                    match_type='phone',
                    threshold=0,
                )
            matches['left_index'].extend(temp1[0])
            matches['right_index'].extend(temp1[1])
            matches['label'].extend(
                ['name/ph mismatch' for i in range(len(temp1[0]))])
                
        case_indices["B"]["ptmt"].append(
            ptmt_group.get_group(key).index[
                ~ptmt_group.get_group(key).index.isin(temp[1] + temp1[1])
            ].to_list()
        )

        case_indices["B"]["fd"].append(
            fd_group.get_group(key).index[
                ~fd_group.get_group(key).index.isin(temp[0] + temp1[0])
            ].to_list()
        )
#date-wise name matching 
idx_l = case_indices["CD"]["fd"]
idx_r = case_indices["CD"]["ptmt"]
    
fd_grp = fd.loc[idx_l].groupby('UPI_Date')
ptmt_grp = ptmt.loc[idx_r].groupby('Transaction_Date_rl')

for key in fd_grp.groups:
    if ptmt_grp.get_group(key)['name_ptmt'].notnull().any():
        temp = match_strings(
            fd_grp.get_group(key)['UPI_Name'],
            ptmt_grp.get_group(key)['name_ptmt'][
                ptmt_grp.get_group(key)['name_ptmt'].notnull()
            ],
            match_type='name',
            threshold=0.37,
        )
    if ptmt_grp.get_group(key)['phone_ptmt'].notnull().any():
        temp = match_strings(
            fd_grp.get_group(key)['Phone'],
            ptmt_grp.get_group(key)['phone_ptmt'][
                ptmt_grp.get_group(key)['phone_ptmt'].notnull()
            ],
            match_type='phone',  # Use match_type 'phone'
            threshold=0.9,       # Use threshold 1 for 'phone' matching
        )
        
    case_indices["C"]["fd"].extend(temp[0])
    case_indices["C"]["ptmt"].extend(temp[1])
        
case_indices["D"]["fd"] = list(set(case_indices["CD"]["fd"])
                                - set(case_indices["C"]["fd"]))
case_indices["D"]["ptmt"] = list(set(case_indices["CD"]["ptmt"])
                                  - set(case_indices["C"]["ptmt"]))
                                  