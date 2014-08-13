import sqlite3
from math import floor, log10
import numpy as np
import pandas as pd
import datetime
from math import sqrt, log

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
    
    def dict_factory(self, row):
        d = {}
        for idx, col in enumerate(self.c.description):
            d[col[0]] = row[idx]
        return d

    def flare_select(self, **kwargs):
        
        if "start" in kwargs:
            start = kwargs['start']
        if "end" in kwargs:
            end = kwargs['end']
        
        self.c.execute('SELECT * FROM flare \
                       WHERE model_peak_time \
                       BETWEEN "'+start+'" AND "'+end+'"\
                       ORDER BY model_peak_time')
        result = self.c.fetchall()
        return result

    def latest(self, **kwargs):

        self.c.execute('SELECT data_peak_time FROM flare ORDER BY \
                       data_peak_time DESC LIMIT 1;')
        result = self.c.fetchall()
        return result[0][0]

    def id_select(self, **kwargs):
        
        if "id" in kwargs:
            start = kwargs['id']
        
        self.c.execute('SELECT * FROM flare \
                       WHERE id = "'+id+'" \
                       ORDER BY model_peak_time')
        result = self.c.fetchall()
        return result

    def flare_dataframe(self, **kwargs):
      if "start" in kwargs:
        start = kwargs['start']
      if "end" in kwargs:
        end = kwargs['end']

      self.c.execute('SELECT * FROM flare WHERE model_peak_time BETWEEN "'+start+'" AND "'+end+'" ORDER BY model_peak_time')
      result = self.c.fetchall()
        
      my_flares = []
      for flare in result: 
        my_flares.append(self.dict_factory(flare))
      my_flares = pd.DataFrame(my_flares)

      times = [datetime.datetime.strptime(a, '%Y-%m-%d %H:%M:%S') for a in my_flares['model_peak_time']]
      startend = [self._flare_boundaries(times[i],
                             my_flares['model_tau_gauss'][i]*3600,
                             my_flares['model_tau_exp'][i]*3600) 
                  for i in range(len(my_flares))]
      start = np.array([i[0] for i in startend])
      end = np.array([i[1] for i in startend])
      length = end-start
      length_sec = [i.total_seconds() for i in length ]
      my_flares['start'] = pd.Series(start, index=my_flares.index)
      my_flares['end'] = pd.Series(end, index=my_flares.index)
      my_flares['length'] = pd.Series(length, index=my_flares.index)
      return my_flares
    
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
            flare = self.dict_factory(flare)
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
        #display(HTML(lines))
        #bottom = HTML("""</tbody></table>""")
        #display(bottom)

        display( HTML(lines + """</tbody></table>""") )

    def _flare_boundaries(self, t0, t_gauss, t_exp, amount = 0.95):
        """

        Calculates the start and end times of a flare given a
            midpoint, a t_gauss value, and a t_exp value.

        Parameters
        ----------
        t0 : datetime.datetime object
            The midpoint time of the flare.
        t_gauss : float
            The gaussian rise parameter. In seconds.
        t_exp : float
            The exponential decay factor. In seconds.
        amount : float
            The amount of the flare which the start and 
            end time should enclose. Defaults to 0.95 (per cent).

        Returns
        -------
        start : datetime.datetime object
            The datetime of the start of the flare.
        end : datetime.datetime object
            The datetime of the end of the flare.

        """

        ramount = 1 - amount

        gauss_cut = sqrt(abs(log(ramount) * 2 * t_gauss**2))
        exp_cut = abs(t_exp*log(ramount)) 
        start = t0 - datetime.timedelta(seconds=gauss_cut)
        end = t0 + datetime.timedelta(seconds=exp_cut)

        return start, end