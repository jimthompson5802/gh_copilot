class MultivariateTestResults:

    def __init__(self, results, endog_names, exog_names):
        self.results = results
        self.endog_names = list(endog_names)
        self.exog_names = list(exog_names)

    def __str__(self):
        return self.summary().__str__()

    def __getitem__(self, item):
        return self.results[item]

    @property
    def summary_frame(self):
        """
        Return results as a multiindex dataframe
        """
        df = []
        for key in self.results:
            tmp = self.results[key]['stat'].copy()
            tmp.loc[:, 'Effect'] = key
            df.append(tmp.reset_index())
        df = pd.concat(df, axis=0)
        df = df.set_index(['Effect', 'index'])
        df.index.set_names(['Effect', 'Statistic'], inplace=True)
        return df

    def summary(self, show_contrast_L=False, show_transform_M=False,
                show_constant_C=False):
        summ = summary2.Summary()
        summ.add_title('Multivariate linear model')
        for key in self.results:
            summ.add_dict({'': ''})
            df = self.results[key]['stat'].copy()
            df = df.reset_index()
            c = list(df.columns)
            c[0] = key
            df.columns = c
            df.index = ['', '', '', '']
            summ.add_df(df)
            if show_contrast_L:
                summ.add_dict({key: ' contrast L='})
                df = pd.DataFrame(self.results[key]['contrast_L'],
                                  columns=self.exog_names)
                summ.add_df(df)
            if show_transform_M:
                summ.add_dict({key: ' transform M='})
                df = pd.DataFrame(self.results[key]['transform_M'],
                                  index=self.endog_names)
                summ.add_df(df)
            if show_constant_C:
                summ.add_dict({key: ' constant C='})
                df = pd.DataFrame(self.results[key]['constant_C'])
                summ.add_df(df)
        return summ