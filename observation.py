import config
from mymysql.mymysql import MyMySQL
from collections import defaultdict, OrderedDict
import matplotlib.pyplot as plt
from ranking.kddcup_searchers import simple_search, get_selected_nodes
from evaluation.kddcup_expt import calc_ground_truth_score
from evaluation.metrics import ndcg2
import json
import numpy as np
from scipy import interpolate
from scipy.optimize import curve_fit
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA


def test_stationarity(timeseries):

    #Determing rolling statistics
    rolmean = pd.rolling_mean(timeseries, window=3)
    rolstd = pd.rolling_std(timeseries, window=3)

    #Plot rolling statistics:
    orig = plt.plot(timeseries.index.values, np.array(timeseries['score']), 'o--', color='blue',label='Original')
    mean = plt.plot(rolmean.index.values, np.array(rolmean['score']), 'o--', color='red', label='Rolling Mean')
    std = plt.plot(rolstd.index.values, np.array(rolstd['score']), 'o--', color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    # plt.show(block=False)

    # #Perform Dickey-Fuller test:
    # print 'Results of Dickey-Fuller Test:'
    # dftest = adfuller(timeseries, autolag='AIC')
    # dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    # for key,value in dftest[4].items():
    #     dfoutput['Critical Value (%s)'%key] = value
    # print dfoutput



rank_2001 = (u'4FF45383', u'0C01DCFD', u'0810AAFA', u'05282E0D', u'05C86094', u'07BAB9E7', u'01124466', u'0011EAC4', u'07874D3C', u'36E5AE3D', u'350410CF', u'050BB43F', u'0619822C', u'00FF56A8', u'000CF342', u'028E8644', u'00A52A7E', u'031C31BE', u'08495949', u'01A8C383', u'06E63D20', u'0A0FF9EB', u'0C34AB7F', u'038AE3AA', u'03FC4F79', u'0A08EC51', u'08EF476D', u'067C4210', u'4D2B1EE6', u'056FA77A', u'0BCE997F', u'04FD6759', u'09EB4F00', u'09B97DFE', u'08802AB2', u'0B0DA3DC', u'06918666', u'06D835C2', u'01F164D0', u'015B46C7', u'01461554', u'025D2C19', u'0A97E0C1', u'022F9467', u'4D8B08D7', u'0B4EC93C', u'09396A59', u'05C31D4B', u'0AD3BA36', u'09B52E07', u'02D66721', u'01198E98', u'01DF6D2D', u'0011114D', u'05669A68', u'0C14EACF', u'08D69EB4', u'86317390', u'043F7F1E', u'04C3E428', u'4D7B0467', u'05DAB8C0', u'0BC22D41', u'0486EC92', u'00F94921', u'07B9F6F5', u'0437D79B', u'01710C78', u'8626355D', u'4CD3CC33', u'03C3607B', u'0349EE07', u'09989277', u'0AE14045', u'03839B48', u'01FB725A', u'06CB2E98', u'03D00B37', u'04F1BEA5', u'06D35AFB', u'0A10A938', u'335ED749', u'0AB1BE73', u'4D68F3D7', u'04BEE6D7', u'05BDC90E', u'00725F06', u'012C9CDF', u'06D18AFD', u'4F076E00', u'0A0E755C', u'09D7E4FA', u'03FCE859', u'08F452CF', u'0ADC5821', u'018A6121', u'0B1B8D25', u'35E9831D', u'03B6E7B2', u'0B08755E', u'0159FE4E', u'01C5C92C', u'0368D319', u'02AA1FA9', u'09EB15AA', u'046C3B83', u'0875EA92', u'06167391', u'86366AC3', u'86263FDB', u'07D2B6B9', u'070DD774', u'00F8422C', u'00FD6FCE', u'009779A9', u'4DDE3B69', u'862979F8', u'0BC0A2DC', u'0815134D', u'0413757D', u'02965656', u'01776B6C', u'004CA7A3', u'09F18F79', u'0757A922', u'0B002509', u'021CC5D8', u'088A680B', u'4E636137', u'35BD5C33', u'05E2161B', u'025139D2', u'0538AC16', u'0966B229', u'05DBB954', u'0C601A7F', u'0474BFAF', u'04B01E55', u'055B394D', u'0507BBD7', u'0CECE932', u'0A4ACFBD', u'002932B0', u'08548E70', u'07162905', u'0C4400FF', u'0B23EC19', u'0296D744', u'0C2DEF79', u'0474FB72')
rank_2002 = (u'4FF45383', u'0C01DCFD', u'00FF56A8', u'06194FDE', u'0815134D', u'083839C7', u'0966B229', u'050BB43F', u'05282E0D', u'4F076E00', u'01776B6C', u'0011EAC4', u'500F4F9C', u'0BF77E37', u'090B77B3', u'0578DF46', u'07511464', u'08F47EBA', u'4E718DAA', u'02D4EDF8', u'00FA1C18', u'0A575E93', u'06601420', u'0A6C6FB4', u'09B48D0B', u'09368EC9', u'0D109F83', u'069F11AE', u'0A97E0C1', u'0619822C', u'350410CF', u'0477FFD3', u'4C7A9B63', u'044C0559', u'000CF342', u'028E8644', u'00A52A7E', u'031C31BE', u'08495949', u'01A8C383', u'06E63D20', u'0A0FF9EB', u'0C34AB7F', u'038AE3AA', u'03FC4F79', u'0A08EC51', u'08EF476D', u'067C4210', u'4D2B1EE6', u'056FA77A', u'0BCE997F', u'04FD6759', u'09EB4F00', u'07BAB9E7', u'09B97DFE', u'08802AB2', u'0B0DA3DC', u'06918666', u'06D835C2', u'01F164D0', u'015B46C7', u'01461554', u'025D2C19', u'022F9467', u'4D8B08D7', u'0B4EC93C', u'09396A59', u'05C31D4B', u'0AD3BA36', u'09B52E07', u'02D66721', u'01198E98', u'01DF6D2D', u'0011114D', u'05669A68', u'0C14EACF', u'08D69EB4', u'86317390', u'043F7F1E', u'04C3E428', u'4D7B0467', u'05DAB8C0', u'0BC22D41', u'0486EC92', u'01124466', u'00F94921', u'07B9F6F5', u'0437D79B', u'01710C78', u'8626355D', u'4CD3CC33', u'03C3607B', u'0349EE07', u'09989277', u'0AE14045', u'03839B48', u'01FB725A', u'06CB2E98', u'03D00B37', u'04F1BEA5', u'06D35AFB', u'0A10A938', u'335ED749', u'0AB1BE73', u'4D68F3D7', u'04BEE6D7', u'05BDC90E', u'00725F06', u'012C9CDF', u'06D18AFD', u'0A0E755C', u'09D7E4FA', u'03FCE859', u'08F452CF', u'0ADC5821', u'018A6121', u'0B1B8D25', u'35E9831D', u'03B6E7B2', u'0B08755E', u'0159FE4E', u'01C5C92C', u'0368D319', u'02AA1FA9', u'09EB15AA', u'046C3B83', u'0875EA92', u'06167391', u'86366AC3', u'86263FDB', u'07D2B6B9', u'070DD774', u'00F8422C', u'00FD6FCE', u'009779A9', u'4DDE3B69', u'862979F8', u'0BC0A2DC', u'0413757D', u'02965656', u'004CA7A3', u'09F18F79', u'0757A922', u'0B002509', u'021CC5D8', u'088A680B', u'4E636137', u'35BD5C33', u'05E2161B', u'025139D2')
rank_2003 = (u'4FF45383', u'05C86094', u'0966B229', u'0A97E0C1', u'0C01DCFD', u'03C436D1', u'0815134D', u'01776B6C', u'08548E70', u'035F81F4', u'09B2D407', u'02AD1047', u'0259891E', u'4E2EC568', u'04BE3F72', u'4EF77785', u'0B185497', u'09B48D0B', u'350410CF', u'045C11B3', u'4C5370B9', u'0D109F83', u'05282E0D', u'0AE14045', u'0011EAC4', u'090B77B3', u'073B94B1', u'0BF6BC56', u'0A2510E5', u'08802AB2', u'044C0559', u'000CF342', u'028E8644', u'00A52A7E', u'031C31BE', u'08495949', u'01A8C383', u'06E63D20', u'0A0FF9EB', u'0C34AB7F', u'038AE3AA', u'03FC4F79', u'0A08EC51', u'08EF476D', u'067C4210', u'4D2B1EE6', u'056FA77A', u'0BCE997F', u'04FD6759', u'09EB4F00', u'07BAB9E7', u'09B97DFE', u'0B0DA3DC', u'06918666', u'06D835C2', u'01F164D0', u'015B46C7', u'01461554', u'025D2C19', u'022F9467', u'4D8B08D7', u'0B4EC93C', u'09396A59', u'05C31D4B', u'0AD3BA36', u'09B52E07', u'02D66721', u'01198E98', u'01DF6D2D', u'0011114D', u'05669A68', u'0C14EACF', u'08D69EB4', u'86317390', u'043F7F1E', u'04C3E428', u'4D7B0467', u'05DAB8C0', u'0BC22D41', u'0486EC92', u'01124466', u'00F94921', u'07B9F6F5', u'0437D79B', u'01710C78', u'8626355D', u'4CD3CC33', u'03C3607B', u'0349EE07', u'09989277', u'03839B48', u'01FB725A', u'06CB2E98', u'03D00B37', u'04F1BEA5', u'06D35AFB', u'0A10A938', u'335ED749', u'0AB1BE73', u'4D68F3D7', u'04BEE6D7', u'05BDC90E', u'00725F06', u'012C9CDF', u'06D18AFD', u'4F076E00', u'0A0E755C', u'09D7E4FA', u'03FCE859', u'08F452CF', u'0ADC5821', u'018A6121', u'0B1B8D25', u'35E9831D', u'03B6E7B2', u'0B08755E', u'0159FE4E', u'01C5C92C', u'0368D319', u'02AA1FA9', u'09EB15AA', u'046C3B83', u'0875EA92', u'06167391', u'86366AC3', u'86263FDB', u'07D2B6B9', u'070DD774', u'00F8422C', u'00FD6FCE', u'009779A9', u'4DDE3B69', u'862979F8', u'0BC0A2DC', u'0413757D', u'02965656', u'004CA7A3', u'09F18F79', u'0757A922', u'0B002509', u'021CC5D8', u'088A680B', u'4E636137', u'35BD5C33', u'05E2161B', u'025139D2', u'0538AC16', u'05DBB954', u'0C601A7F', u'0474BFAF')
rank_2004 = (u'4FF45383', u'0966B229', u'05C86094', u'05282E0D', u'01776B6C', u'09299093', u'0259891E', u'04BE3F72', u'0A2FAFA5', u'09B48D0B', u'4C6E8511', u'350410CF', u'0D109F83', u'03C436D1', u'0BF77E37', u'3333E44B', u'04BEE6D7', u'0A2510E5', u'4D0FB79F', u'07511464', u'4E2EC568', u'4D634C00', u'07CB626B', u'0B003FA6', u'0679E020', u'03496C27', u'4F10C1EB', u'0527EA39', u'004B1FBC', u'01124466', u'0134B592', u'017C99DB', u'0619822C', u'09988D7C', u'0349EE07', u'000CF342', u'028E8644', u'00A52A7E', u'031C31BE', u'08495949', u'01A8C383', u'06E63D20', u'0A0FF9EB', u'0C34AB7F', u'038AE3AA', u'03FC4F79', u'0A08EC51', u'08EF476D', u'067C4210', u'4D2B1EE6', u'056FA77A', u'0BCE997F', u'04FD6759', u'09EB4F00', u'07BAB9E7', u'09B97DFE', u'08802AB2', u'0B0DA3DC', u'06918666', u'06D835C2', u'01F164D0', u'015B46C7', u'01461554', u'025D2C19', u'0A97E0C1', u'022F9467', u'4D8B08D7', u'0B4EC93C', u'09396A59', u'05C31D4B', u'0AD3BA36', u'09B52E07', u'02D66721', u'01198E98', u'01DF6D2D', u'0011114D', u'05669A68', u'0C14EACF', u'08D69EB4', u'86317390', u'043F7F1E', u'04C3E428', u'4D7B0467', u'05DAB8C0', u'0BC22D41', u'0486EC92', u'00F94921', u'07B9F6F5', u'0437D79B', u'01710C78', u'8626355D', u'4CD3CC33', u'03C3607B', u'09989277', u'0AE14045', u'03839B48', u'01FB725A', u'06CB2E98', u'03D00B37', u'04F1BEA5', u'06D35AFB', u'0A10A938', u'335ED749', u'0AB1BE73', u'4D68F3D7', u'05BDC90E', u'00725F06', u'012C9CDF', u'06D18AFD', u'4F076E00', u'0A0E755C', u'09D7E4FA', u'03FCE859', u'08F452CF', u'0ADC5821', u'018A6121', u'0B1B8D25', u'35E9831D', u'03B6E7B2', u'0B08755E', u'0159FE4E', u'01C5C92C', u'0368D319', u'02AA1FA9', u'09EB15AA', u'046C3B83', u'0875EA92', u'06167391', u'86366AC3', u'86263FDB', u'07D2B6B9', u'070DD774', u'00F8422C', u'00FD6FCE', u'009779A9', u'4DDE3B69', u'862979F8', u'0BC0A2DC', u'0815134D', u'0413757D', u'02965656', u'004CA7A3', u'09F18F79', u'0011EAC4', u'0757A922', u'0B002509', u'021CC5D8', u'088A680B', u'4E636137', u'35BD5C33')
rank_2005 = (u'4FF45383', u'0966B229', u'017C99DB', u'350410CF', u'01776B6C', u'08E4D2D6', u'4E1BB800', u'0BC7337B', u'0810AAFA', u'05282E0D', u'012C9CDF', u'09D7E4FA', u'02C7065C', u'099D876D', u'0112E226', u'00C50601', u'00D4F325', u'04BE3F72', u'07874D3C', u'0C01DCFD', u'4ECD74BC', u'0477FFD3', u'07CB626B', u'07B6A2A8', u'06601420', u'0754B26A', u'0B46E8A6', u'066A71BC', u'004B1FBC', u'00FF56A8', u'4EF77785', u'0045D019', u'0011EAC4', u'05C86094', u'035F81F4', u'0A2FAFA5', u'000CF342', u'028E8644', u'00A52A7E', u'031C31BE', u'08495949', u'01A8C383', u'06E63D20', u'0A0FF9EB', u'0C34AB7F', u'038AE3AA', u'03FC4F79', u'0A08EC51', u'08EF476D', u'067C4210', u'4D2B1EE6', u'056FA77A', u'0BCE997F', u'04FD6759', u'09EB4F00', u'07BAB9E7', u'09B97DFE', u'08802AB2', u'0B0DA3DC', u'06918666', u'06D835C2', u'01F164D0', u'015B46C7', u'01461554', u'025D2C19', u'0A97E0C1', u'022F9467', u'4D8B08D7', u'0B4EC93C', u'09396A59', u'05C31D4B', u'0AD3BA36', u'09B52E07', u'02D66721', u'01198E98', u'01DF6D2D', u'0011114D', u'05669A68', u'0C14EACF', u'08D69EB4', u'86317390', u'043F7F1E', u'04C3E428', u'4D7B0467', u'05DAB8C0', u'0BC22D41', u'0486EC92', u'01124466', u'00F94921', u'07B9F6F5', u'0437D79B', u'01710C78', u'8626355D', u'4CD3CC33', u'03C3607B', u'0349EE07', u'09989277', u'0AE14045', u'03839B48', u'01FB725A', u'06CB2E98', u'03D00B37', u'04F1BEA5', u'06D35AFB', u'0A10A938', u'335ED749', u'0AB1BE73', u'4D68F3D7', u'04BEE6D7', u'05BDC90E', u'00725F06', u'06D18AFD', u'4F076E00', u'0A0E755C', u'03FCE859', u'08F452CF', u'0ADC5821', u'018A6121', u'0B1B8D25', u'35E9831D', u'03B6E7B2', u'0B08755E', u'0159FE4E', u'01C5C92C', u'0368D319', u'02AA1FA9', u'09EB15AA', u'046C3B83', u'0875EA92', u'06167391', u'86366AC3', u'86263FDB', u'07D2B6B9', u'070DD774', u'00F8422C', u'00FD6FCE', u'009779A9', u'4DDE3B69', u'862979F8', u'0BC0A2DC', u'0815134D', u'0413757D', u'02965656', u'004CA7A3', u'09F18F79', u'0757A922', u'0B002509', u'021CC5D8', u'088A680B', u'4E636137')
rank_2006 = (u'4FF45383', u'0966B229', u'05282E0D', u'017C99DB', u'4F076E00', u'031C31BE', u'01776B6C', u'0BF77E37', u'0259891E', u'0011114D', u'4EF77785', u'00FF56A8', u'06CBC4BA', u'0C37CBAD', u'0134B592', u'0A97E0C1', u'09396A59', u'01124466', u'500F4F9C', u'06194FDE', u'0B78A521', u'099D876D', u'0C1EB600', u'09299093', u'4D0FB79F', u'08E4D2D6', u'0810AAFA', u'0A5DAC76', u'09E1988B', u'07CB626B', u'09DD720A', u'06D39D72', u'091BC727', u'081E3F30', u'0312B01D', u'045716B5', u'348EB203', u'007D2F41', u'004B1FBC', u'08D7E515', u'004CA7A3', u'09B2D407', u'04BE3F72', u'08F47EBA', u'0A2FAFA5', u'4CE6FC2D', u'0003B055', u'09988D7C', u'01C5C92C', u'035F81F4', u'04F0A8A0', u'350410CF', u'0BC7337B', u'4E2EC568', u'000CF342', u'028E8644', u'00A52A7E', u'08495949', u'01A8C383', u'06E63D20', u'0A0FF9EB', u'0C34AB7F', u'038AE3AA', u'03FC4F79', u'0A08EC51', u'08EF476D', u'067C4210', u'4D2B1EE6', u'056FA77A', u'0BCE997F', u'04FD6759', u'09EB4F00', u'07BAB9E7', u'09B97DFE', u'08802AB2', u'0B0DA3DC', u'06918666', u'06D835C2', u'01F164D0', u'015B46C7', u'01461554', u'025D2C19', u'022F9467', u'4D8B08D7', u'0B4EC93C', u'05C31D4B', u'0AD3BA36', u'09B52E07', u'02D66721', u'01198E98', u'01DF6D2D', u'05669A68', u'0C14EACF', u'08D69EB4', u'86317390', u'043F7F1E', u'04C3E428', u'4D7B0467', u'05DAB8C0', u'0BC22D41', u'0486EC92', u'00F94921', u'07B9F6F5', u'0437D79B', u'01710C78', u'8626355D', u'4CD3CC33', u'03C3607B', u'0349EE07', u'09989277', u'0AE14045', u'03839B48', u'01FB725A', u'06CB2E98', u'03D00B37', u'04F1BEA5', u'06D35AFB', u'0A10A938', u'335ED749', u'0AB1BE73', u'4D68F3D7', u'04BEE6D7', u'05BDC90E', u'00725F06', u'012C9CDF', u'06D18AFD', u'0A0E755C', u'09D7E4FA', u'03FCE859', u'08F452CF', u'0ADC5821', u'018A6121', u'0B1B8D25', u'35E9831D', u'03B6E7B2', u'0B08755E', u'0159FE4E', u'0368D319', u'02AA1FA9', u'09EB15AA', u'046C3B83', u'0875EA92', u'06167391', u'86366AC3', u'86263FDB', u'07D2B6B9', u'070DD774', u'00F8422C', u'00FD6FCE', u'009779A9')
rank_2007 = (u'4FF45383', u'08548E70', u'0966B229', u'01776B6C', u'06194FDE', u'09E1988B', u'04F0A8A0', u'0BF77E37', u'0C37CBAD', u'0134B592', u'004B1FBC', u'099D876D', u'012C9CDF', u'0810AAFA', u'031C31BE', u'08EF476D', u'0A97E0C1', u'01198E98', u'4F076E00', u'09D7E4FA', u'00F8422C', u'0011EAC4', u'05C86094', u'035F81F4', u'03C436D1', u'0B78A521', u'4D0FB79F', u'090B77B3', u'00C50601', u'0073ECCC', u'0259891E', u'4D634C00', u'00F9B76E', u'0C01DCFD', u'4EF77785', u'08D6D550', u'4F1AE805', u'05B3DFF1', u'074491E2', u'0B46E8A6', u'05282E0D', u'066A71BC', u'01C641C3', u'045716B5', u'01A17328', u'0A254897', u'0154426A', u'017C99DB', u'348EB203', u'023B9EF3', u'04B5445A', u'00FF56A8', u'06CBC4BA', u'08D69EB4', u'038AE3AA', u'026778A2', u'350410CF', u'0045D019', u'0B659C39', u'4C6E8511', u'000CF342', u'028E8644', u'00A52A7E', u'08495949', u'01A8C383', u'06E63D20', u'0A0FF9EB', u'0C34AB7F', u'03FC4F79', u'0A08EC51', u'067C4210', u'4D2B1EE6', u'056FA77A', u'0BCE997F', u'04FD6759', u'09EB4F00', u'07BAB9E7', u'09B97DFE', u'08802AB2', u'0B0DA3DC', u'06918666', u'06D835C2', u'01F164D0', u'015B46C7', u'01461554', u'025D2C19', u'022F9467', u'4D8B08D7', u'0B4EC93C', u'09396A59', u'05C31D4B', u'0AD3BA36', u'09B52E07', u'02D66721', u'01DF6D2D', u'0011114D', u'05669A68', u'0C14EACF', u'86317390', u'043F7F1E', u'04C3E428', u'4D7B0467', u'05DAB8C0', u'0BC22D41', u'0486EC92', u'01124466', u'00F94921', u'07B9F6F5', u'0437D79B', u'01710C78', u'8626355D', u'4CD3CC33', u'03C3607B', u'0349EE07', u'09989277', u'0AE14045', u'03839B48', u'01FB725A', u'06CB2E98', u'03D00B37', u'04F1BEA5', u'06D35AFB', u'0A10A938', u'335ED749', u'0AB1BE73', u'4D68F3D7', u'04BEE6D7', u'05BDC90E', u'00725F06', u'06D18AFD', u'0A0E755C', u'03FCE859', u'08F452CF', u'0ADC5821', u'018A6121', u'0B1B8D25', u'35E9831D', u'03B6E7B2', u'0B08755E', u'0159FE4E', u'01C5C92C', u'0368D319', u'02AA1FA9', u'09EB15AA', u'046C3B83', u'0875EA92', u'06167391', u'86366AC3', u'86263FDB', u'07D2B6B9')
rank_2008 = (u'4FF45383', u'0966B229', u'00FF56A8', u'0578DF46', u'0259891E', u'05C86094', u'0B46E8A6', u'035F81F4', u'4EF77785', u'05B3DFF1', u'05282E0D', u'08D7E515', u'07CA5884', u'4F076E00', u'031C31BE', u'0BC22D41', u'03839B48', u'09EA1A37', u'06194FDE', u'0BF77E37', u'0749A315', u'3333E44B', u'09B2D407', u'0B00BA67', u'02AD1047', u'08E4D2D6', u'4D634C00', u'0C01DCFD', u'073B94B1', u'0134B592', u'09E1988B', u'0619822C', u'0B003FA6', u'081E3F30', u'01C641C3', u'017C99DB', u'09368EC9', u'0B659C39', u'080DBBEF', u'012C9CDF', u'08EF476D', u'350410CF', u'335ED749', u'0B46561F', u'4C5370B9', u'004B1FBC', u'0C1EB600', u'0045D9F6', u'00D956B2', u'4F10C1EB', u'03C0079C', u'000CF342', u'028E8644', u'00A52A7E', u'08495949', u'01A8C383', u'06E63D20', u'0A0FF9EB', u'0C34AB7F', u'038AE3AA', u'03FC4F79', u'0A08EC51', u'067C4210', u'4D2B1EE6', u'056FA77A', u'0BCE997F', u'04FD6759', u'09EB4F00', u'07BAB9E7', u'09B97DFE', u'08802AB2', u'0B0DA3DC', u'06918666', u'06D835C2', u'01F164D0', u'015B46C7', u'01461554', u'025D2C19', u'0A97E0C1', u'022F9467', u'4D8B08D7', u'0B4EC93C', u'09396A59', u'05C31D4B', u'0AD3BA36', u'09B52E07', u'02D66721', u'01198E98', u'01DF6D2D', u'0011114D', u'05669A68', u'0C14EACF', u'08D69EB4', u'86317390', u'043F7F1E', u'04C3E428', u'4D7B0467', u'05DAB8C0', u'0486EC92', u'01124466', u'00F94921', u'07B9F6F5', u'0437D79B', u'01710C78', u'8626355D', u'4CD3CC33', u'03C3607B', u'0349EE07', u'09989277', u'0AE14045', u'01FB725A', u'06CB2E98', u'03D00B37', u'04F1BEA5', u'06D35AFB', u'0A10A938', u'0AB1BE73', u'4D68F3D7', u'04BEE6D7', u'05BDC90E', u'00725F06', u'06D18AFD', u'0A0E755C', u'09D7E4FA', u'03FCE859', u'08F452CF', u'0ADC5821', u'018A6121', u'0B1B8D25', u'35E9831D', u'03B6E7B2', u'0B08755E', u'0159FE4E', u'01C5C92C', u'0368D319', u'02AA1FA9', u'09EB15AA', u'046C3B83', u'0875EA92', u'06167391', u'86366AC3', u'86263FDB', u'07D2B6B9', u'070DD774', u'00F8422C', u'00FD6FCE', u'009779A9', u'4DDE3B69', u'862979F8', u'0BC0A2DC')
rank_2009 = (u'4FF45383', u'0966B229', u'066A71BC', u'0259891E', u'0134B592', u'0BF77E37', u'3333E44B', u'4C6E8511', u'08EF476D', u'08E4D2D6', u'4E1BB800', u'09E1988B', u'00D956B2', u'0C37CBAD', u'004B1FBC', u'00FF56A8', u'0C01DCFD', u'0A97E0C1', u'099D876D', u'012C9CDF', u'069F11AE', u'01A8C383', u'081E3F30', u'0ACF7BFE', u'04F1BEA5', u'04BEE6D7', u'0368D319', u'0815134D', u'01776B6C', u'05C86094', u'035F81F4', u'4E045540', u'0B00BA67', u'0C1EB600', u'0228E4F2', u'090B77B3', u'07CA5884', u'04BE3F72', u'01F6ADD2', u'03C0079C', u'003E7116', u'00BC8D07', u'0108DAB7', u'0477FFD3', u'08A948CC', u'04281A41', u'045716B5', u'017C99DB', u'0AC1438B', u'04B5445A', u'0D109F83', u'0B890E85', u'0B659C39', u'06CBC4BA', u'026778A2', u'074491E2', u'09368EC9', u'500F4F9C', u'05E1B1D9', u'00C50601', u'0A5DAC76', u'0626251D', u'06194FDE', u'0413757D', u'09D7E4FA', u'4F05DC4B', u'00FA1C18', u'0B46E8A6', u'050BB43F', u'09B52E07', u'031AFA6D', u'04946B1E', u'4CE6FC2D', u'023B9EF3', u'012BCF09', u'01FACAD8', u'085ADB71', u'09E3EE34', u'0A254897', u'044C0559', u'000CF342', u'028E8644', u'00A52A7E', u'031C31BE', u'08495949', u'06E63D20', u'0A0FF9EB', u'0C34AB7F', u'038AE3AA', u'03FC4F79', u'0A08EC51', u'067C4210', u'4D2B1EE6', u'056FA77A', u'0BCE997F', u'04FD6759', u'09EB4F00', u'07BAB9E7', u'09B97DFE', u'08802AB2', u'0B0DA3DC', u'06918666', u'06D835C2', u'01F164D0', u'015B46C7', u'01461554', u'025D2C19', u'022F9467', u'4D8B08D7', u'0B4EC93C', u'09396A59', u'05C31D4B', u'0AD3BA36', u'02D66721', u'01198E98', u'01DF6D2D', u'0011114D', u'05669A68', u'0C14EACF', u'08D69EB4', u'86317390', u'043F7F1E', u'04C3E428', u'4D7B0467', u'05DAB8C0', u'0BC22D41', u'0486EC92', u'01124466', u'00F94921', u'07B9F6F5', u'0437D79B', u'01710C78', u'8626355D', u'4CD3CC33', u'03C3607B', u'0349EE07', u'09989277', u'0AE14045', u'03839B48', u'01FB725A', u'06CB2E98', u'03D00B37', u'06D35AFB', u'0A10A938', u'335ED749', u'0AB1BE73', u'4D68F3D7', u'05BDC90E', u'00725F06', u'06D18AFD')
rank_2010 = (u'4FF45383', u'0966B229', u'066A71BC', u'0259891E', u'0134B592', u'0BF77E37', u'3333E44B', u'4C6E8511', u'08EF476D', u'08E4D2D6', u'0C37CBAD', u'4E1BB800', u'09E1988B', u'00D956B2', u'004B1FBC', u'00FF56A8', u'0C01DCFD', u'0A97E0C1', u'099D876D', u'012C9CDF', u'069F11AE', u'01A8C383', u'081E3F30', u'0ACF7BFE', u'04F1BEA5', u'04BEE6D7', u'0368D319', u'0815134D', u'01776B6C', u'05C86094', u'035F81F4', u'4E045540', u'0B00BA67', u'0C1EB600', u'0228E4F2', u'090B77B3', u'07CA5884', u'04BE3F72', u'01F6ADD2', u'03C0079C', u'003E7116', u'00BC8D07', u'0108DAB7', u'0477FFD3', u'08A948CC', u'04281A41', u'045716B5', u'017C99DB', u'0AC1438B', u'04B5445A', u'0D109F83', u'0B890E85', u'0B659C39', u'06CBC4BA', u'026778A2', u'074491E2', u'09368EC9', u'500F4F9C', u'05E1B1D9', u'00C50601', u'0A5DAC76', u'0626251D', u'06194FDE', u'0413757D', u'09D7E4FA', u'4F05DC4B', u'00FA1C18', u'0B46E8A6', u'050BB43F', u'09B52E07', u'031AFA6D', u'04946B1E', u'4CE6FC2D', u'023B9EF3', u'012BCF09', u'01FACAD8', u'085ADB71', u'09E3EE34', u'0A254897', u'044C0559', u'000CF342', u'028E8644', u'00A52A7E', u'031C31BE', u'08495949', u'06E63D20', u'0A0FF9EB', u'0C34AB7F', u'038AE3AA', u'03FC4F79', u'0A08EC51', u'067C4210', u'4D2B1EE6', u'056FA77A', u'0BCE997F', u'04FD6759', u'09EB4F00', u'07BAB9E7', u'09B97DFE', u'08802AB2', u'0B0DA3DC', u'06918666', u'06D835C2', u'01F164D0', u'015B46C7', u'01461554', u'025D2C19', u'022F9467', u'4D8B08D7', u'0B4EC93C', u'09396A59', u'05C31D4B', u'0AD3BA36', u'02D66721', u'01198E98', u'01DF6D2D', u'0011114D', u'05669A68', u'0C14EACF', u'08D69EB4', u'86317390', u'043F7F1E', u'04C3E428', u'4D7B0467', u'05DAB8C0', u'0BC22D41', u'0486EC92', u'01124466', u'00F94921', u'07B9F6F5', u'0437D79B', u'01710C78', u'8626355D', u'4CD3CC33', u'03C3607B', u'0349EE07', u'09989277', u'0AE14045', u'03839B48', u'01FB725A', u'06CB2E98', u'03D00B37', u'06D35AFB', u'0A10A938', u'335ED749', u'0AB1BE73', u'4D68F3D7', u'05BDC90E', u'00725F06', u'06D18AFD')
rank_2011 = (u'4FF45383', u'0966B229', u'0259891E', u'05C86094', u'08E4D2D6', u'0228E4F2', u'0B0ADEB6', u'4F4B6B43', u'08A948CC', u'09988D7C', u'012C9CDF', u'031C31BE', u'0B185497', u'4C6E8511', u'06194FDE', u'026778A2', u'0754B26A', u'00FF56A8', u'3333E44B', u'08EF476D', u'0D109F83', u'01124466', u'4F076E00', u'087CBF68', u'0BF77E37', u'00A3C9C2', u'083839C7', u'04592826', u'07CA5884', u'08F47EBA', u'0C37CBAD', u'0762929E', u'4E1BB800', u'4F10C1EB', u'0B46E8A6', u'05282E0D', u'0626251D', u'081E3F30', u'09368EC9', u'06CBC4BA', u'04AF492D', u'0AE14045', u'00C50601', u'01A17328', u'004B1FBC', u'0011EAC4', u'4C5370B9', u'0134B592', u'4D71CB46', u'0A2510E5', u'0368E8BE', u'0B3B54E4', u'025F5CB1', u'36E5AE3D', u'4E718DAA', u'09B48D0B', u'050BB43F', u'0B659C39', u'069F11AE', u'01A8C383', u'0A97E0C1', u'08802AB2', u'4DADEF94', u'02AD1047', u'0092C1B8', u'073B94B1', u'0BC7337B', u'000CF342', u'028E8644', u'00A52A7E', u'08495949', u'06E63D20', u'0A0FF9EB', u'0C34AB7F', u'038AE3AA', u'03FC4F79', u'0A08EC51', u'067C4210', u'4D2B1EE6', u'056FA77A', u'0BCE997F', u'04FD6759', u'09EB4F00', u'07BAB9E7', u'09B97DFE', u'0B0DA3DC', u'06918666', u'06D835C2', u'01F164D0', u'015B46C7', u'01461554', u'025D2C19', u'022F9467', u'4D8B08D7', u'0B4EC93C', u'09396A59', u'05C31D4B', u'0AD3BA36', u'09B52E07', u'02D66721', u'01198E98', u'01DF6D2D', u'0011114D', u'05669A68', u'0C14EACF', u'08D69EB4', u'86317390', u'043F7F1E', u'04C3E428', u'4D7B0467', u'05DAB8C0', u'0BC22D41', u'0486EC92', u'00F94921', u'07B9F6F5', u'0437D79B', u'01710C78', u'8626355D', u'4CD3CC33', u'03C3607B', u'0349EE07', u'09989277', u'03839B48', u'01FB725A', u'06CB2E98', u'03D00B37', u'04F1BEA5', u'06D35AFB', u'0A10A938', u'335ED749', u'0AB1BE73', u'4D68F3D7', u'04BEE6D7', u'05BDC90E', u'00725F06', u'06D18AFD', u'0A0E755C', u'09D7E4FA', u'03FCE859', u'08F452CF', u'0ADC5821', u'018A6121', u'0B1B8D25', u'35E9831D', u'03B6E7B2', u'0B08755E', u'0159FE4E', u'01C5C92C', u'0368D319', u'02AA1FA9')
rank_2012 = (u'4FF45383', u'05C86094', u'07CA5884', u'0966B229', u'069F11AE', u'0D109F83', u'0259891E', u'0A2FAFA5', u'0BF77E37', u'0626251D', u'0154426A', u'4F076E00', u'004B1FBC', u'4C6E8511', u'08E4D2D6', u'0C01DCFD', u'06194FDE', u'0011114D', u'01124466', u'03C436D1', u'09B2D407', u'04BE3F72', u'863BFDE1', u'08F7F83A', u'0134B592', u'09E1988B', u'05282E0D', u'066A71BC', u'050BB43F', u'0535FE32', u'348EB203', u'06CBC4BA', u'00FF56A8', u'028E8644', u'0578DF46', u'09368EC9', u'07D2B6B9', u'012C9CDF', u'026778A2', u'0BC2EB17', u'0754B26A', u'04D98949', u'00A52A7E', u'01FE29EA', u'06D27A8D', u'00F9B76E', u'4EF77785', u'0A567B93', u'0A2510E5', u'0810AAFA', u'03C0079C', u'4F10C1EB', u'081E3F30', u'4F1AE805', u'0B46E8A6', u'01F1D439', u'0312B01D', u'4CE6FC2D', u'0003B055', u'4F05DC4B', u'0056B275', u'00FA1C18', u'4EBC7FCF', u'000CF342', u'031C31BE', u'08495949', u'01A8C383', u'06E63D20', u'0A0FF9EB', u'0C34AB7F', u'038AE3AA', u'03FC4F79', u'0A08EC51', u'08EF476D', u'067C4210', u'4D2B1EE6', u'056FA77A', u'0BCE997F', u'04FD6759', u'09EB4F00', u'07BAB9E7', u'09B97DFE', u'08802AB2', u'0B0DA3DC', u'06918666', u'06D835C2', u'01F164D0', u'015B46C7', u'01461554', u'025D2C19', u'0A97E0C1', u'022F9467', u'4D8B08D7', u'0B4EC93C', u'09396A59', u'05C31D4B', u'0AD3BA36', u'09B52E07', u'02D66721', u'01198E98', u'01DF6D2D', u'05669A68', u'0C14EACF', u'08D69EB4', u'86317390', u'043F7F1E', u'04C3E428', u'4D7B0467', u'05DAB8C0', u'0BC22D41', u'0486EC92', u'00F94921', u'07B9F6F5', u'0437D79B', u'01710C78', u'8626355D', u'4CD3CC33', u'03C3607B', u'0349EE07', u'09989277', u'0AE14045', u'03839B48', u'01FB725A', u'06CB2E98', u'03D00B37', u'04F1BEA5', u'06D35AFB', u'0A10A938', u'335ED749', u'0AB1BE73', u'4D68F3D7', u'04BEE6D7', u'05BDC90E', u'00725F06', u'06D18AFD', u'0A0E755C', u'09D7E4FA', u'03FCE859', u'08F452CF', u'0ADC5821', u'018A6121', u'0B1B8D25', u'35E9831D', u'03B6E7B2', u'0B08755E', u'0159FE4E', u'01C5C92C', u'0368D319', u'02AA1FA9', u'09EB15AA')
rank_2013 = (u'4FF45383', u'0966B229', u'0BF77E37', u'0259891E', u'862ADA3F', u'05C86094', u'863BFDE1', u'09368EC9', u'06194FDE', u'07CA5884', u'0B46E8A6', u'00FF56A8', u'06CBC4BA', u'0134B592', u'4E718DAA', u'0A97E0C1', u'01124466', u'04BEE6D7', u'0011EAC4', u'34DF872C', u'4DFEDD28', u'083839C7', u'0A2510E5', u'0B0ADEB6', u'04BE3F72', u'4EF77785', u'026778A2', u'09E1988B', u'4F1AE805', u'0653F611', u'4F10C1EB', u'050BB43F', u'04F0A8A0', u'007D2F41', u'004B1FBC', u'0D109F83', u'069F11AE', u'0762929E', u'099D876D', u'0A4ACFBD', u'09DC92C5', u'0578DF46', u'017C99DB', u'03839B48', u'06041575', u'0C01DCFD', u'01A8C383', u'0154426A', u'09EA1A37', u'01FE29EA', u'4E2EC568', u'07930A8B', u'05282E0D', u'4CC89935', u'00C50601', u'0B4EC93C', u'0815134D', u'3333E44B', u'04B5445A', u'4CDD5BAE', u'0A0E755C', u'0C4400FF', u'000CF342', u'028E8644', u'00A52A7E', u'031C31BE', u'08495949', u'06E63D20', u'0A0FF9EB', u'0C34AB7F', u'038AE3AA', u'03FC4F79', u'0A08EC51', u'08EF476D', u'067C4210', u'4D2B1EE6', u'056FA77A', u'0BCE997F', u'04FD6759', u'09EB4F00', u'07BAB9E7', u'09B97DFE', u'08802AB2', u'0B0DA3DC', u'06918666', u'06D835C2', u'01F164D0', u'015B46C7', u'01461554', u'025D2C19', u'022F9467', u'4D8B08D7', u'09396A59', u'05C31D4B', u'0AD3BA36', u'09B52E07', u'02D66721', u'01198E98', u'01DF6D2D', u'0011114D', u'05669A68', u'0C14EACF', u'08D69EB4', u'86317390', u'043F7F1E', u'04C3E428', u'4D7B0467', u'05DAB8C0', u'0BC22D41', u'0486EC92', u'00F94921', u'07B9F6F5', u'0437D79B', u'01710C78', u'8626355D', u'4CD3CC33', u'03C3607B', u'0349EE07', u'09989277', u'0AE14045', u'01FB725A', u'06CB2E98', u'03D00B37', u'04F1BEA5', u'06D35AFB', u'0A10A938', u'335ED749', u'0AB1BE73', u'4D68F3D7', u'05BDC90E', u'00725F06', u'012C9CDF', u'06D18AFD', u'4F076E00', u'09D7E4FA', u'03FCE859', u'08F452CF', u'0ADC5821', u'018A6121', u'0B1B8D25', u'35E9831D', u'03B6E7B2', u'0B08755E', u'0159FE4E', u'01C5C92C', u'0368D319', u'02AA1FA9', u'09EB15AA', u'046C3B83', u'0875EA92')
rank_2014 = (u'4FF45383', u'0011EAC4', u'0D109F83', u'0BF77E37', u'07CA5884', u'05C86094', u'026778A2', u'0C01DCFD', u'05282E0D', u'0966B229', u'862ADA3F', u'01A8C383', u'0259891E', u'86317390', u'069F11AE', u'00FF56A8', u'0B46E8A6', u'00C50601', u'348EB203', u'08EF476D', u'4F05DC4B', u'350410CF', u'06CBC4BA', u'038CB6F6', u'09368EC9', u'0A97E0C1', u'01198E98', u'0AB1BE73', u'01776B6C', u'34DF872C', u'035F81F4', u'0749A315', u'3333E44B', u'0A2510E5', u'0B00BA67', u'00784876', u'07511464', u'03C0079C', u'863BFDE1', u'0BC2EB17', u'4ECD74BC', u'09E1988B', u'0B908117', u'09B48D0B', u'056EA0E2', u'4F10C1EB', u'09988D7C', u'081E3F30', u'066A71BC', u'023B9EF3', u'06194FDE', u'03839B48', u'0B0ADEB6', u'04F0A8A0', u'08D7E515', u'00F8422C', u'099D876D', u'07874D3C', u'4EF77785', u'0699B629', u'0619822C', u'080DBBEF', u'09EB4F00', u'03C436D1', u'0A2FAFA5', u'022B685F', u'4EBC7FCF', u'000CF342', u'028E8644', u'00A52A7E', u'031C31BE', u'08495949', u'06E63D20', u'0A0FF9EB', u'0C34AB7F', u'038AE3AA', u'03FC4F79', u'0A08EC51', u'067C4210', u'4D2B1EE6', u'056FA77A', u'0BCE997F', u'04FD6759', u'07BAB9E7', u'09B97DFE', u'08802AB2', u'0B0DA3DC', u'06918666', u'06D835C2', u'01F164D0', u'015B46C7', u'01461554', u'025D2C19', u'022F9467', u'4D8B08D7', u'0B4EC93C', u'09396A59', u'05C31D4B', u'0AD3BA36', u'09B52E07', u'02D66721', u'01DF6D2D', u'0011114D', u'05669A68', u'0C14EACF', u'08D69EB4', u'043F7F1E', u'04C3E428', u'4D7B0467', u'05DAB8C0', u'0BC22D41', u'0486EC92', u'01124466', u'00F94921', u'07B9F6F5', u'0437D79B', u'01710C78', u'8626355D', u'4CD3CC33', u'03C3607B', u'0349EE07', u'09989277', u'0AE14045', u'01FB725A', u'06CB2E98', u'03D00B37', u'04F1BEA5', u'06D35AFB', u'0A10A938', u'335ED749', u'4D68F3D7', u'04BEE6D7', u'05BDC90E', u'00725F06', u'012C9CDF', u'06D18AFD', u'4F076E00', u'0A0E755C', u'09D7E4FA', u'03FCE859', u'08F452CF', u'0ADC5821', u'018A6121', u'0B1B8D25', u'35E9831D', u'03B6E7B2', u'0B08755E', u'0159FE4E', u'01C5C92C', u'0368D319')
rank_2015 = (u'0966B229', u'05C86094', u'01A8C383', u'00FF56A8', u'4FF45383', u'0D109F83', u'07CA5884', u'05282E0D', u'0B00BA67', u'06CBC4BA', u'03C436D1', u'0259891E', u'056FA77A', u'01776B6C', u'0BF77E37', u'00C50601', u'081E3F30', u'0653F611', u'026778A2', u'4C6E8511', u'04D98949', u'3333E44B', u'0154426A', u'0134B592', u'09D7E4FA', u'0A97E0C1', u'0BC22D41', u'01124466', u'03839B48', u'04BEE6D7', u'0011EAC4', u'8623AC36', u'34DF872C', u'09DDAA88', u'373A6845', u'06194FDE', u'0C1EB600', u'04592826', u'0B7D39A1', u'500B3463', u'0C01DCFD', u'03FC3F40', u'0762929E', u'0B46E8A6', u'0352694C', u'09630970', u'038CB6F6', u'069F11AE', u'4E718DAA', u'01461554', u'0A08EC51', u'031C31BE', u'069BF47D', u'066A71BC', u'03E298B8', u'08EF476D', u'350410CF', u'09368EC9', u'050BB43F', u'862ADA3F', u'04F0A8A0', u'0B0ADEB6', u'0A2FAFA5', u'0699B629', u'061C6C3B', u'09EB4F00', u'4F10C1EB', u'08D7E515', u'0BC2EB17', u'08802AB2', u'08548E70', u'0BC3491B', u'4F4B6B43', u'0045D019', u'09988D7C', u'045716B5', u'05EFFB90', u'07874D3C', u'0B317C57', u'0A6C6FB4', u'36E22876', u'348EB203', u'00BC8D07', u'000CF342', u'028E8644', u'00A52A7E', u'08495949', u'06E63D20', u'0A0FF9EB', u'0C34AB7F', u'038AE3AA', u'03FC4F79', u'067C4210', u'4D2B1EE6', u'0BCE997F', u'04FD6759', u'07BAB9E7', u'09B97DFE', u'0B0DA3DC', u'06918666')
rank_avg1_4 = (u'4FF45383', u'0966B229', u'05C86094', u'0259891E', u'0BF77E37', u'0D109F83', u'07CA5884', u'0011EAC4', u'069F11AE', u'00FF56A8', u'026778A2', u'862ADA3F', u'0B46E8A6', u'05282E0D', u'06CBC4BA', u'0C01DCFD', u'06194FDE', u'09368EC9', u'863BFDE1', u'01A8C383', u'4F10C1EB', u'0B0ADEB6', u'00C50601', u'09E1988B', u'348EB203', u'0A2510E5', u'004B1FBC', u'08E4D2D6', u'0134B592', u'09988D7C', u'08EF476D', u'0A2FAFA5', u'0A97E0C1', u'01124466', u'86317390', u'0626251D', u'3333E44B', u'4C6E8511', u'34DF872C', u'0154426A', u'081E3F30', u'050BB43F', u'4F076E00', u'066A71BC', u'4EF77785', u'4F05DC4B', u'350410CF', u'038CB6F6', u'04BE3F72', u'04F0A8A0', u'012C9CDF', u'0BC2EB17', u'0762929E', u'083839C7', u'09B48D0B', u'03C0079C', u'03839B48', u'099D876D', u'4E718DAA', u'0754B26A', u'0578DF46', u'0228E4F2', u'4F4B6B43', u'08A948CC', u'01198E98', u'0AB1BE73', u'01776B6C', u'035F81F4', u'0749A315', u'0B00BA67', u'00784876', u'07511464', u'4ECD74BC', u'0B908117', u'056EA0E2', u'023B9EF3', u'4F1AE805', u'031C31BE', u'0B185497', u'03C436D1', u'04BEE6D7', u'4DFEDD28', u'0653F611', u'007D2F41', u'01FE29EA', u'0011114D', u'09B2D407', u'08F7F83A', u'0535FE32', u'08D7E515', u'0A4ACFBD', u'09DC92C5', u'017C99DB', u'087CBF68', u'00A3C9C2', u'04592826', u'08F47EBA', u'0C37CBAD', u'4E1BB800', u'04AF492D')

ranks = [rank_2001, rank_2002, rank_2003, rank_2004, rank_2005, rank_2006, rank_2007, rank_2008, rank_2009, rank_2010, rank_2011, rank_2012, rank_2013, rank_2014, rank_2015, rank_avg1_4]

watching_affil = (u'0966B229', u'05C86094', u'01A8C383', u'00FF56A8', u'4FF45383', u'0D109F83', u'07CA5884', u'05282E0D', u'0B00BA67', u'06CBC4BA', u'03C436D1', u'0259891E', u'056FA77A', u'01776B6C', u'0BF77E37', u'00C50601', u'081E3F30', u'0653F611', u'026778A2', u'4C6E8511', u'04D98949', u'3333E44B', u'0154426A', u'0134B592', u'09D7E4FA', u'0A97E0C1', u'0BC22D41', u'01124466', u'03839B48', u'04BEE6D7', u'0011EAC4', u'8623AC36', u'34DF872C', u'09DDAA88', u'373A6845', u'06194FDE', u'0C1EB600', u'04592826', u'0B7D39A1', u'500B3463', u'0C01DCFD', u'03FC3F40', u'0762929E', u'0B46E8A6', u'0352694C', u'09630970', u'038CB6F6', u'069F11AE', u'4E718DAA', u'01461554', u'0A08EC51', u'031C31BE', u'069BF47D', u'066A71BC', u'03E298B8', u'08EF476D', u'350410CF', u'09368EC9', u'050BB43F', u'862ADA3F', u'04F0A8A0', u'0B0ADEB6', u'0A2FAFA5', u'0699B629', u'061C6C3B', u'09EB4F00', u'4F10C1EB', u'08D7E515', u'0BC2EB17', u'08802AB2', u'08548E70', u'0BC3491B', u'4F4B6B43', u'0045D019', u'09988D7C', u'045716B5', u'05EFFB90', u'07874D3C', u'0B317C57', u'0A6C6FB4', u'36E22876', u'348EB203', u'00BC8D07', u'000CF342', u'028E8644', u'00A52A7E', u'08495949', u'06E63D20', u'0A0FF9EB', u'0C34AB7F', u'038AE3AA', u'03FC4F79', u'067C4210', u'4D2B1EE6', u'0BCE997F', u'04FD6759', u'07BAB9E7', u'09B97DFE', u'0B0DA3DC', u'06918666')

# rank_trends = defaultdict(dict)

# for i in range(0, 6):
#     for k, affil in enumerate(ranks[i]):
#         rank_trends[affil][i] = k


# for k, v in rank_trends.items():
#     plt.figure()
#     plt.title(k)
#     plt.plot(v.keys(), v.values(), 'o--')
#     # plt.show()
#     plt.savefig('img/%s.png' % k)



# for i in range(4):
    # print len(set(ranks[i][:80]) & set(ranks[i+1][:20]))
# print len(set(rank_2010[:100]) & set(rank_2011[:20]))

db = MyMySQL(db=config.DB_NAME, user=config.DB_USER, passwd=config.DB_PASSWD)

selected_affils = db.select(fields="id", table="selected_affils")

def tsa_pred():
    affil_scores_trends = OrderedDict()

    with open('affil_scores_trends.json', 'r') as fp:
        affil_scores_trends = json.load(fp)
        fp.close()

    # for year in range(2011, 2016):
    #     results = simple_search(selected_affils, 'KDD', [year], expand_year=[], age_decay=False, age_relev=0.0)
    #     results = dict(results)

    #     for affil in watching_affil:
    #         try:
    #             affil_scores_trends[affil][year] = results[affil]
    #         except:
    #             affil_scores_trends[affil] = {year: results[affil]}

    # for year in range(2001, 2011):
    #     results = simple_search(selected_affils, 'KDD', [], expand_year=[year], age_decay=False, age_relev=0.0)
    #     results = dict(results)

    #     for affil in watching_affil:
    #         try:
    #             affil_scores_trends[affil][year] = results[affil]
    #         except:
    #             affil_scores_trends[affil] = {year: results[affil]}


    # with open('affil_scores_trends.json', 'w') as fp:
    #     json.dump(affil_scores_trends, fp)
    #     fp.close()

    # import pdb;pdb.set_trace()

    pred_affil_score = defaultdict()
    dateindex = pd.DatetimeIndex(freq='12m', start='2001', periods=14)
    for affil in watching_affil[:80]:
        record = affil_scores_trends[affil]
        record = sorted(record.items(), key=lambda d:d[0])[:14]
        x = [int(year) for year in zip(*record)[0]]
        y = zip(*record)[1]
        # plt.figure()
        # plt.title('KDD - %s'%affil)
        # plt.plot(x, y, 'o--', label='Original')

        # # interploat
        # spl = interpolate.UnivariateSpline(x[:-1], y[:-1])
        # # spl = interpolate.InterpolatedUnivariateSpline(x[:-1], y[:-1])
        # xs = np.linspace(x[0], x[-1], 100)
        # plt.plot(xs, spl(xs), 'g', lw=3)

        df = pd.DataFrame(np.array(y), index=dateindex, columns=['score'])
        # rolmean = pd.rolling_mean(df, window=2)
        # plt.plot(np.array(x), rolmean, color='red', label='Rolling mean')
        # rol_avg_diff = (np.array(y[1:]) - np.array(rolmean.dropna()).reshape((len(x)-1,)))

        # plt.plot(np.array(x[1:]), rol_avg_diff, color='blue', label='Rolling avg diff')
        # decomposition = seasonal_decompose(df)
        # trend = decomposition.trend
        # seasonal = decomposition.seasonal
        # residual = decomposition.resid

        # plt.subplot(411)
        # plt.plot(df, label='Original')
        # plt.legend(loc='best')
        # plt.subplot(412)
        # plt.plot(trend, label='Trend')
        # plt.legend(loc='best')
        # plt.subplot(413)
        # plt.plot(seasonal,label='Seasonality')
        # plt.legend(loc='best')
        # plt.subplot(414)
        # plt.plot(residual, label='Residuals')
        # plt.legend(loc='best')
        # plt.tight_layout()
        # # test_stationarity(df)
        # plt.savefig('img/TSA/decomp/KDD-%s.png'%affil)


        # #ACF and PACF plots:
        # lag_acf = acf(df, nlags=3)
        # lag_pacf = pacf(df, nlags=3, method='ols')

        # #Plot ACF:
        # plt.subplot(121)
        # plt.plot(lag_acf)
        # plt.axhline(y=0,linestyle='--',color='gray')
        # plt.axhline(y=-1.96/np.sqrt(len(df)),linestyle='--',color='gray')
        # plt.axhline(y=1.96/np.sqrt(len(df)),linestyle='--',color='gray')
        # plt.title('Autocorrelation Function')

        # #Plot PACF:
        # plt.subplot(122)
        # plt.plot(lag_pacf)
        # plt.axhline(y=0,linestyle='--',color='gray')
        # plt.axhline(y=-1.96/np.sqrt(len(df)),linestyle='--',color='gray')
        # plt.axhline(y=1.96/np.sqrt(len(df)),linestyle='--',color='gray')
        # plt.title('Partial Autocorrelation Function')
        # plt.tight_layout()

        # # AR model
        # plt.subplot(311)
        # model = ARIMA(df, order=(1, 1, 0))
        # results_AR = model.fit(disp=-1)
        # plt.plot(df, label='Original')
        # plt.plot(results_AR.fittedvalues, color='red', label='AR model')
        # plt.title('RSS: %.4f'% np.sum((results_AR.fittedvalues-df[1:])**2)['score'])
        # plt.legend(loc='best')

        # # MA model
        # plt.subplot(312)
        # model = ARIMA(df, order=(1, 1, 0))
        # results_MA = model.fit(disp=-1)
        # plt.plot(df, label='Original')
        # plt.plot(results_MA.fittedvalues, color='red', label='MA model')
        # plt.title('RSS: %.4f'% np.sum((results_MA.fittedvalues-df[1:])**2)['score'])
        # plt.legend(loc='best')

        # combined model
        # plt.subplot(313)
        try:
            model = ARIMA(df, order=(1, 1, 0))
            results_ARIMA = model.fit(disp=-1)
        except Exception, e:
            print e
            # import pdb;pdb.set_trace()
            continue
        # plt.plot(df, label='Original')
        # plt.plot(results_ARIMA.fittedvalues, color='red', label='ARIMA')
        # plt.title('RSS: %.4f'% np.sum((results_ARIMA.fittedvalues-df[1:])**2)['score'])
        # plt.legend(loc='best')

        predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy=True)
        # print predictions_ARIMA_diff.head()
        predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
        predictions_ARIMA =  pd.Series(np.array([df.ix[0]['score'] for i in range(len(df.index))]), index=df.index)
        predictions_ARIMA = predictions_ARIMA.add(predictions_ARIMA_diff_cumsum,fill_value=0)
        # plt.plot(df, label='Original')
        # plt.plot(predictions_ARIMA)
        # plt.title('RMSE: %.4f'% np.sqrt(np.sum((predictions_ARIMA-df)**2)['score']/len(df)))
        # plt.legend(loc='best')

        # predict
        pred_diff = results_ARIMA.predict('2015-01-31','2015-01-31', dynamic = True)
        # pred_diff = results_ARIMA.predict('2016-01-31','2018-01-31', dynamic = True)
        pred_diff_cumsum = pred_diff.cumsum()
        preds =  pd.Series(np.array([predictions_ARIMA['2014-01-31'] for i in range(1)]), index=pd.DatetimeIndex(freq='12m', start='2015', periods=1))
        preds = preds.add(pred_diff_cumsum)
        # plt.plot(pd.concat([predictions_ARIMA, preds]), label='Predicted')
        # plt.legend(loc='best')
        # plt.show()
        # plt.savefig('img/TSA/pred/KDD-%s.png'%affil)

        pred_affil_score[affil] = preds['2015-01-31']

    return pred_affil_score



if __name__ == '__main__':
    pred_affil_score = tsa_pred()
    print pred_affil_score
    # import pdb;pdb.set_trace()
    results = get_selected_nodes(pred_affil_score, selected_affils)
    ground_truth = calc_ground_truth_score(selected_affils, 'KDD')
    actual, relevs = zip(*ground_truth)
    pred = zip(*results)[0]
    ndcg = ndcg2(actual, pred, relevs, k=20)
    print "NDCG@20: %s" % ndcg

