import pandas as pd
import requests 
import datetime
from datetime import timedelta
from datetime import datetime
import math
import numpy as np
import pulp
from pulp import LpVariable
from pulp import LpProblem
from pulp import LpMaximize, lpSum
import json




#1863591
class Optimiser:

    def __init__(self,id,weeks):
        self.id = id
        self.weeks = weeks
        self.r = self.getBootstrap()
        self.team_names = self.getTeams()
        self.gameweek = self.getCurrentGameweek()
        
        

    def run(self,N):
        
        
        self.squad = self.getFirstSquad()
        self.getChipEvents()
        self.updateSquad()
        self.getPlayerDf()
        self.getLatestGamesData()
        self.mergePlayersPoints()
        self.mergeSquadPointsInfo()
        self.calcSquadSellingCost()
        self.updateDataValue()
        
        self.getBank()
        self.getBudget()
        self.getDataColumnsForOptimiser()
        self.getUniqueSquadPlayerNames()
        
        self.prob = LpProblem("FPL Player Choices", LpMaximize)
        #variables
        self.playerVariables()
        self.captainVariables()
        self.main11Variables()
        

        #set objective
        self.prob.setObjective(self.objectiveFunction())

        
        #constraints
        self.constraintBudget()
        self.constraintPositions()
        self.constraintNChanges(N)
        self.constraint3PerClub()
        self.constraintSquadSize()
        self.constraint1Captain()
        self.constraintCaptainSelected()
        self.constraintmain11()
        self.constraintMain11Selected()
        self.constraintMain11Positions()




        self.prob.solve()

        self.results()


        changes = self.findChanges()
        captain = self.getCaptain()
        
        print(changes)
        print(captain)
        
        #return changes.to_html(classes='table table-stripped')
        #return changes["Name"].iloc[0]#, changes["Change"].iloc[0]
        return changes

    def best15(self):
        self.getPlayerDf()
        self.getLatestGamesData()
        self.mergePlayersPoints()

        self.getDataColumnsForOptimiser()
        
        
        self.data["value"] = self.data["now_cost"]
        self.prob = LpProblem("FPL Player Choices", LpMaximize)
        self.playerVariables()
        self.captainVariables()
        self.main11Variables()

        #set objective
        self.prob.setObjective(self.objectiveFunction())

        self.BUDGET = 1000
        #constraints
        self.constraintBudget()
        self.constraintPositions()
        
        self.constraint3PerClub()
        self.constraintSquadSize()
        self.constraint1Captain()
        self.constraintCaptainSelected()
        self.constraintmain11()
        self.constraintMain11Positions()
        self.constraintMain11Selected()


        self.prob.solve()
        self.results()

        captain = self.getCaptain()
        print(self.team)

        players_df = self.players_df.rename(columns = {"name":"Name","id":"ID"})
        merged_df = players_df.merge(self.team, on='Name', how='inner', indicator=True)
        merged_df = merged_df.rename(columns = {"team_name":"Team"})

        return merged_df





    def getFirstSquad(self):

        url = "https://fantasy.premierleague.com/api/entry/"+str(self.id)+"/event/1/picks/"
        response = requests.get(url)
        r = response.json()
        picks = pd.json_normalize(r["picks"])

        picks = picks["element"].sort_values().reset_index(drop = True)

        prices = []
        for pick in picks:
            url =  "https://fantasy.premierleague.com/api/element-summary/"+str(pick)+"/"

            response = requests.get(url)
            r = response.json()
            game1 =  pd.json_normalize(r["history"])
    
            prices.append(game1["value"][0])

        prices = pd.DataFrame(prices)
        squad = pd.concat([picks,prices],axis=1)
        squad.columns = ["element","purchase_price"]
        squad = squad.set_index("element")
        return squad
    


    def getChipEvents(self):
        history_url = "https://fantasy.premierleague.com/api/entry/"+str(self.id)+"/history/"
        response = requests.get(history_url)
        r = response.json()
        chips = pd.json_normalize(r["chips"])
        try:
            chips = chips.set_index("name")   
            freehit = chips['event'].loc["freehit"]
        except:
            freehit = None

        
        self.freehit_week = freehit



    def updateSquad(self):
        url = "https://fantasy.premierleague.com/api/entry/"+ str(self.id) +"/transfers/"
        response = requests.get(url)
        r = response.json()


        
        ## update squad list and costs

        for i in range(len(r)):
            
            if self.freehit_week != r[-i-1]["event"]:
                #add added players
                new_player =pd.DataFrame({"element":r[-i-1]["element_in"],"purchase_price":r[-i-1]["element_in_cost"]} ,index = [0])
                new_player = new_player.set_index("element")
                self.squad = pd.concat([self.squad,new_player])
                #drop removed players
                self.squad = self.squad.drop( r[-i-1]["element_out"],axis = 0)

        self.squad = self.squad.reset_index()
        
    

    def getBootstrap(self):
        url = "https://fantasy.premierleague.com/api/bootstrap-static/"
        response = requests.get(url)
        r = response.json()
        return r
        


    def getCurrentGameweek(self):
        today = datetime.now().timestamp()
        fixtures_df = pd.json_normalize(self.r['events']) 

        try:
            fixtures_df = fixtures_df.loc[fixtures_df.deadline_time_epoch>today]
            gameweek =  fixtures_df.iloc[0].id
        except:
            gameweek =  fixtures_df.iloc[-1].id
        return gameweek
    
    def getTeams(self):
        teams_data = pd.json_normalize(self.r["teams"])
        team_names = teams_data[["id","name"]]
        team_names = team_names.rename(columns = {"id":"team","name":"team_name"})

        return team_names
    
    def getTeamStrength(self):
        teams_data = pd.json_normalize(self.r["teams"])
        teams_data = teams_data[["id","name","short_name","points","position","strength_overall_home","strength_overall_away"]]
        return teams_data

    def getPlayerDf(self):
        players_df=pd.json_normalize(self.r['elements']) 

        players_df['name'] = players_df['first_name'] + ' ' + players_df['second_name']

        mapping = {
            1: 'GK',
            2: 'DEF',
            3: 'MID',
            4: 'FWD'
        }
        
        players_df['position'] =  players_df['element_type'].map(mapping)
        players_df = players_df[['id',"name" ,'team',"chance_of_playing_next_round","chance_of_playing_this_round","position","now_cost"]]

        players_df = pd.merge(players_df,self.team_names,on = "team")

        players_df["chance_of_playing_next_round"] = players_df["chance_of_playing_next_round"].fillna(100)
        players_df["chance_of_playing_this_round"] = players_df["chance_of_playing_this_round"].fillna(0)

        self.players_df = players_df
    
    def getLatestGamesData(self):

        try:
            latest_points = []
            for gw in range(max(self.gameweek-self.weeks,1),self.gameweek):
                url = "https://fantasy.premierleague.com/api/event/"+str(gw)+"/live/"
                response = requests.get(url)
                r = response.json()
                
                temp = pd.json_normalize(r['elements'])
                        
                latest_points.append(temp)
            
            latest_points = pd.concat(latest_points)
            latest_points = latest_points[["id","stats.total_points"]]
            latest_points = latest_points.rename(columns={"stats.total_points":"points"})
            latest_points = latest_points.groupby("id").sum()

        except:
            latest_points_df = pd.json_normalize(self.r['elements'])
            latest_points = latest_points_df[["id","total_points"]]
            latest_points = latest_points.rename(columns={"total_points":"points"})
        


        self.latest_points = latest_points
        
    
    def mergePlayersPoints(self):
        ## merge with pos current value, team, name and points last 5 games
        data = pd.merge(self.players_df,self.latest_points, on = "id",how = "inner")
        data["points"]=data["points"]*data["chance_of_playing_next_round"]/100
        
        self.data = data

    def mergeSquadPointsInfo(self):
        squad = pd.merge(self.squad,self.latest_points,left_on = "element",right_on ="id")
        squad = pd.merge(squad,self.players_df,left_on = "element",right_on ="id",how="inner")
        self.squad = squad


    def calcSellDiff(self,row):

        if row["now_cost"]<=row["purchase_price"]:
            return row["now_cost"]
        if row["now_cost"]>row["purchase_price"]:
            return math.floor(row["purchase_price"]+( row["now_cost"]-row["purchase_price"] )/ 2)


    def calcSquadSellingCost(self):
        ## calculate selling cost
        # Apply the function to each row
        self.squad["selling_price"] = self.squad.apply(self.calcSellDiff, axis=1)

    def updateDataValue(self):
        merged_data = self.data.merge(self.squad[['id', 'selling_price']], on='id', how='left')
        self.data['now_cost'] = np.where(~merged_data['selling_price'].isnull(), merged_data['selling_price'], merged_data['now_cost'])



    def getBank(self):
        url = "https://fantasy.premierleague.com/api/entry/"+str(self.id)+"/"
        response = requests.get(url)
        r = response.json()
        self.bank = r["last_deadline_bank"]

    def getBudget(self):
        self.BUDGET = sum(self.squad["selling_price"]) + self.bank


    def getUniqueSquadPlayerNames(self):
        self.NAMES = self.squad["name"].unique()

    def getDataColumnsForOptimiser(self):
        self.names = self.data["name"]
        self.teams = self.data["team"]
        self.positions = self.data["position"]
        self.prices = self.data["now_cost"]
        self.points = self.data["points"]
        self.POS = self.data["position"].unique()
        self.CLUBS = self.data["team"].unique()

    def playerVariables(self):
        self.players = [LpVariable("player_" + str(i), cat="Binary") for i in self.data.index]

    def captainVariables(self):
        self.captain = [LpVariable("Captain_" + str(i), cat="Binary") for i in  self.data.index]

    def main11Variables(self):
        self.main11 = [LpVariable("mainplayer_" + str(i), cat="Binary") for i in  self.data.index]

    def objectiveFunction(self):
        obj = lpSum([(self.players[i] * self.points[i]) + (self.captain[i] * self.points[i]) + (self.main11[i] * self.points[i]) for i in range(len(self.data))])
        return obj
    
    def constraintBudget(self):
        budget_constraint = pulp.LpConstraint(
            e = lpSum(self.players[i] * self.data.now_cost[self.data.index[i]] for i in range(len(self.data))) ,
            rhs = self.BUDGET,
            sense = pulp.LpConstraintLE,
            name = "budget_constraint" )
        self.prob.addConstraint(budget_constraint)

    def constraintPositions(self):

        pos_available = {
        'DEF': 5,
        'FWD': 3,
        'MID': 5,
        'GK': 2,}

        for pos in self.POS:
            pos_constraint = pulp.LpConstraint(
                e = lpSum(self.players[i] for i in range(len(self.data)) if self.positions[i] == pos)  ,
                rhs = pos_available[pos],
            sense = pulp.LpConstraintLE,
            name = f"pos_constraint{pos}" )
            
            self.prob.addConstraint(pos_constraint)

    def constraintMain11Positions(self):

        pos_available = {
        'DEF': 5,
        'FWD': 3,
        'MID': 5,
        'GK': 1,}

        for pos in self.POS:
            pos_constraint = pulp.LpConstraint(
                e = lpSum(self.main11[i] for i in range(len(self.data)) if self.positions[i] == pos)  ,
                rhs = pos_available[pos],
            sense = pulp.LpConstraintLE,
            name = f"pos_main11_constraint{pos}" )
            
            self.prob.addConstraint(pos_constraint)

    def constraintNChanges(self,N):
        constraint2changes = pulp.LpConstraint(
            e =  lpSum(self.players[i] for i in range(len(self.data)) if self.names[i] in self.NAMES),
            rhs = 15-int(N),
            sense = pulp.LpConstraintGE,
            name = "constraint2changes")
        self.prob.addConstraint(constraint2changes)

    def constraint3PerClub(self):
        club_constraints = []
        for club in self.CLUBS:
            constraint_club = pulp.LpConstraint( 
                e=lpSum(self.players[i] for i in range(len(self.data)) if self.teams[i] == club),
                rhs=3,
                sense=pulp.LpConstraintLE,
                name=f"constraint_club_{club}")
            club_constraints.append(constraint_club)
        self.prob.extend(club_constraints)


    def constraintSquadSize(self):
        constraint_squad_size = pulp.LpConstraint(
            e = lpSum(self.players[i] for i in range(len(self.data))),
            rhs = 15,
            sense = pulp.LpConstraintEQ,
            name = "constraintSquadSize")
        self.prob.addConstraint(constraint_squad_size)

    def constraint1Captain(self):

        constraint_captain = pulp.LpConstraint(
            e = lpSum(self.captain[i] for i in range(len(self.data))),
            rhs = 1,
            sense = pulp.LpConstraintEQ,
            name = "constraintcaptain")
        self.prob.addConstraint(constraint_captain)

    def constraintCaptainSelected(self):

        for i in range(len(self.data)):
            self.prob += self.captain[i] <= self.players[i]     

    def constraintmain11(self):

        constraint_main11 = pulp.LpConstraint(
            e = lpSum(self.main11[i] for i in range(len(self.data))),
            rhs = 11,
            sense = pulp.LpConstraintEQ,
            name = "constraintmain11")
        self.prob.addConstraint(constraint_main11)

    def constraintMain11Selected(self):

        for i in range(len(self.data)):
            self.prob += self.main11[i] <= self.players[i] 

    def results(self):
        results = []

        for p in self.main11:
            if p.varValue != 0:
                index = int(p.name.split("_")[1])
                name = self.names[index]
                print(name,index,p.varValue)

        for v in self.players:
            if v.varValue != 0:
                index = int(v.name.split("_")[1])
                name = self.names[index]
                club = self.teams[index]
                position = self.positions[index]
                point = self.points[index]
                price = self.data.now_cost[index]
                results.append([name, position, club, point, price])

        # Create a DataFrame from the results list
        df_results = pd.DataFrame(results, columns=['Name', 'Position', 'Club', 'Total_Points', 'Price'])
  
        team = df_results[['Name','Position','Club','Total_Points','Price']]
        team = team.sort_values("Name")
        #print(team)
        self.team = team



    def select_value(row):
        if pd.isnull(row['column_with_null']):
            return row['column_to_select_from']
        else:
            return row['column_with_null']
        
    def findChanges(self):
        squad = self.squad.rename(columns = {"name":"Name"})
        merged_df = squad.merge(self.team, on='Name', how='outer', indicator=True)

        # Filter the rows that are different
        changes = merged_df[merged_df['_merge'] != 'both']
        #print(changes.columns)
        changes = changes[["Name","_merge","selling_price",'Price','team_name',"Club"]]
        changes = changes.merge(self.team_names,how='left', left_on = "Club",right_on = "team" )
        #changes["Team"] = changes[]
        
        changes["Team"] = np.where(changes['_merge'] == 'left_only', changes["team_name_x"],changes["team_name_y"])
        changes['_merge'] = np.where(changes['_merge'] == 'left_only', 'out', 'in')
        changes = changes.rename(columns = {"_merge":"Change"})
        changes = changes.fillna("")
        #print(changes)
        return changes

    def getCaptain(self):
        captain_name = []
        for v in self.captain:
            if v.varValue != 0:

                index = int(v.name.split("_")[1])
                name = self.data.name[index]
                captain_name.append([name])

        captain_name = pd.DataFrame(captain_name)

        print(captain_name[0].values)
        return captain_name





    
            

    


    









#1863591
model = Optimiser(8636416,9)

print(model.run(3))
#print(model.best15())
