from flask import Flask, render_template, redirect, url_for
import requests
import pandas as pd
import pulp
import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_team')
def generate_team():
    try:
        # Step 1: Fetch data from FPL API
        url = "https://fantasy.premierleague.com/api/bootstrap-static/"
        response = requests.get(url, timeout=10)  # Added timeout
        response.raise_for_status()  # Raise HTTP errors if any
        data = response.json()
        players = data['elements']
        df = pd.DataFrame(players)
        df_available = df[(df['status'] == 'a') | (df['status'] == 'd')]
        
        # Step 2: Process and filter data
        goalkeepers = df_available[df_available['element_type'] == 1].sort_values(by='ep_next', ascending=False).head(20)
        defenders = df_available[df_available['element_type'] == 2].sort_values(by='ep_next', ascending=False).head(20)
        midfielders = df_available[df_available['element_type'] == 3].sort_values(by='ep_next', ascending=False).head(20)
        forwards = df_available[df_available['element_type'] == 4].sort_values(by='ep_next', ascending=False).head(20)

        merged_df = pd.concat([goalkeepers, defenders, midfielders, forwards], ignore_index=True)
        merged_df = merged_df[['first_name', 'second_name', 'web_name', 'team', 'now_cost', 'ep_next', 'element_type', 'total_points', 'event_points']]
        merged_df.to_csv('players.csv', index=False, encoding='utf-8-sig')

        # Step 3: Load data from CSV and solve optimization problem
        df = pd.read_csv("players.csv")
        prob = pulp.LpProblem("FPL_Optimal_XI", pulp.LpMaximize)
        player_vars = pulp.LpVariable.dicts("Player", df.index, cat='Binary')
        prob += pulp.lpSum(df.loc[i, 'ep_next'] * player_vars[i] for i in df.index)
        
        # Constraints
        prob += pulp.lpSum(player_vars[i] for i in df.index) == 11
        prob += pulp.lpSum(player_vars[i] for i in df[df['element_type'] == 1].index) == 1  # 1 GK
        prob += pulp.lpSum(player_vars[i] for i in df[df['element_type'] == 2].index) >= 3  # Min 3 DEF
        prob += pulp.lpSum(player_vars[i] for i in df[df['element_type'] == 2].index) <= 5  # Max 5 DEF
        prob += pulp.lpSum(player_vars[i] for i in df[df['element_type'] == 3].index) >= 2  # Min 2 MID
        prob += pulp.lpSum(player_vars[i] for i in df[df['element_type'] == 3].index) <= 5  # Max 5 MID
        prob += pulp.lpSum(player_vars[i] for i in df[df['element_type'] == 4].index) >= 1  # Min 1 FWD
        prob += pulp.lpSum(player_vars[i] for i in df[df['element_type'] == 4].index) <= 3  # Max 3 FWD
        prob += pulp.lpSum(df.loc[i, 'now_cost'] * player_vars[i] for i in df.index) <= 8500  # Budget constraint
        
        # Max 3 players per team
        for team in df['team'].unique():
            prob += pulp.lpSum(player_vars[i] for i in df[df['team'] == team].index) <= 3

        prob.solve()

        if pulp.LpStatus[prob.status] == 'Optimal':
            selected_indices = [i for i in df.index if player_vars[i].varValue == 1]
            selected_team = df.loc[selected_indices]
            total_cost = selected_team['now_cost'].sum()
            total_points = selected_team['ep_next'].sum()

            # Get captain and vice-captain
            captain_vise = selected_team.sort_values(by="ep_next", ascending=False).head(2)
            captain = captain_vise.iloc[0]['web_name']
            vice_captain = captain_vise.iloc[1]['web_name']

            # Map element_type to position names
            position_map = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}
            selected_team['position'] = selected_team['element_type'].map(position_map)

            return render_template('index.html', 
                                 team=selected_team.to_dict('records'), 
                                 total_cost=total_cost, 
                                 total_points=total_points,
                                 captain=captain, 
                                 vice_captain=vice_captain)
        else:
            return render_template('index.html', error="No feasible solution found. Consider relaxing constraints.")
    
    except requests.RequestException as e:
        return render_template('index.html', error=f"Failed to fetch FPL data: {str(e)}")
    except Exception as e:
        return render_template('index.html', error=f"An error occurred: {str(e)}")

if __name__ == '__main__':
    # app.run(debug=True)
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))