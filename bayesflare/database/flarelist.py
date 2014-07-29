import sqlite3
from math import floor, log10
import numpy as np

class Flare_List():
    
    def __init__(self,filename):
        self.conn = sqlite3.connect(filename)
        self.c = self.conn.cursor()
        
    def setup_flare_table(self):
        self.c.execute('''
          CREATE TABLE flare
            (
              id INTEGER PRIMARY KEY ASC, 
              thresh_start_time DATETIME NOT NULL,
              data_peak_time DATETIME NOT NULL,
              data_peak_amplitude DOUBLE NOT NULL,
              thresh_end_time DATETIME NOT NULL,
              
              model_tau_gauss DOUBLE NOT NULL,
              model_tau_exp DOUBLE NOT NULL,
              model_peak_time DATETIME NOT NULL,
              model_peak_amplitude DOUBLE NOT NULL,
              
              flags TEXT
            )
          ''')

    def commit(self):
        self.conn.commit()
        
    def close(self):
        self.conn.commit()
        self.conn.close()
        
    def save_flare(self, 
                   threshold_start_time, 
                   threshold_peak_time,
                   data_peak_amplitude,
                   threshold_end_time, 
                   model_tau_gauss,
                   model_tau_exp,
                   model_peak_time,
                   model_peak_amplitude,
                   flags="NULL"
                   ):
        
        database = self.c
        
        threshold_start_time = str(threshold_start_time)
        threshold_end_time = str(threshold_end_time)
        threshold_peak_time = str(threshold_peak_time)
        data_peak_amplitude = str(data_peak_amplitude)
        model_tau_gauss = str(model_tau_gauss)
        model_tau_exp = str(model_tau_exp)
        model_peak_time = str(model_peak_time)
        model_peak_amplitude = str(model_peak_amplitude)
        flags = str(flags)

        database.execute('''
            INSERT INTO flare VALUES
            (
                NULL,"''' +
                threshold_start_time+'''","'''+
                threshold_peak_time+'''","'''+
                data_peak_amplitude+'''","'''+ 
                threshold_end_time+'''","'''+ 
                model_tau_gauss+'''","'''+ 
                model_tau_exp+'''","'''+ 
                model_peak_time+'''","'''+  
                model_peak_amplitude+'''","'''+
                flags+'''" 
            )
        ''')
        self.commit()
        
    def goes_class(self, energy):
        if energy == 'nan':
            return 'Z'
        energye = int(floor(log10(energy)))
        energies = np.array([-9, -8, -7, -6, -5, -4, -3, -2])
        categories = ['Below A', 'A', 'B', 'C', 'M', 'X', 'Above X']
        subcat= floor(10*(energy/10**floor(log10(energy))))/10
        return categories[np.where(energies==energye)[0][0]]+str(subcat)
    
    def dict_factory(self, cursor, row):
        d = {}
        for idx, col in enumerate(cursor.description):
            d[col[0]] = row[idx]
        return d
    
    def flare_table(self, **kwargs):
        
        if "start" in kwargs:
            start = kwargs['start']
        if "end" in kwargs:
            end = kwargs['end']
        
        self.c.execute('SELECT * FROM flare WHERE model_peak_time BETWEEN "'+start+'" AND "'+end+'" ORDER BY model_peak_time')
        result = self.c.fetchall()
        
        from IPython.display import HTML, Javascript, display
        lines = """
                <h2>BayesFlare Flare Listing</h2>
                <table class="table table-striped">
                    <thead>
                    <tr>
                        <th> DBID </th>
                        <th> Start </th>
                        <th> Class </th>
                        <th> Peak </th>
                        <th> End </th>
                        <th> Model Peak </th>
                        <th> Model Amp </th>
                        <th> Model TauE </th>
                        <th> Model TauG </th>
                    </tr>
                    </thead>
                    <tbody class="table table-striped table-condensed">
                """
        for i in range(len(result)):
            flare = result[i]
            flare = self.dict_factory(self.c, flare)
            line = """
                <tr>
                    <td>%s</td>
                    <td>%s</td>
                    <td>%s</td>
                    <td>%s</td>
                    <td>%s</td>
                    <td>%s</td>
                    <td>%s</td>
                    <td>%f</td>
                    <td>%f</td>
                </tr>
                """%(
                    flare['id'],
                    flare['thresh_start_time'], 
                    self.goes_class(flare['data_peak_amplitude']),
                    flare['data_peak_time'],
                    flare['thresh_end_time'],
                    flare['model_peak_time'],
                    self.goes_class(flare['model_peak_amplitude']),
                    flare['model_tau_exp'],
                    flare['model_tau_gauss']
                  )
            lines = lines + line
        display(HTML(lines))
        bottom = HTML("""</tbody></table>""")
        display(bottom)