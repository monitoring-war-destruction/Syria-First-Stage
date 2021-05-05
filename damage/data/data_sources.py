
DATA_SOURCES = {
    'aleppo': {
        'annotations': [
            '6_Damage_Sites_Aleppo_SDA.shp',
        ],
        'rasters': [
            #'aleppo_2011_01_01_zoom_19.tif',
            'aleppo_2011_06_26_zoom_19.tif', # Too bright for prewar
            'aleppo_2013_05_26_zoom_19.tif',
            'aleppo_2013_09_23_zoom_19.tif',
            'aleppo_2013_10_31_zoom_19.tif',
            'aleppo_2014_01_31_zoom_19.tif',
            #'aleppo_2014_05_23_zoom_19.tif', # Cloudy
            'aleppo_2014_07_14_zoom_19.tif',
            'aleppo_2014_10_22_zoom_19.tif',
            #'aleppo_2014_11_07_zoom_19.tif', # Corrupted
            'aleppo_2014_12_15_zoom_19.tif',
            'aleppo_2015_10_26_zoom_19.tif',
            'aleppo_2015_11_22_zoom_19.tif',
            'aleppo_2015_12_11_zoom_19.tif',
            'aleppo_2016_03_23_zoom_19.tif',
            'aleppo_2016_03_29_zoom_19.tif',
            'aleppo_2016_04_06_zoom_19.tif',
	    'aleppo_2016_06_12_zoom_19.tif',
            'aleppo_2016_07_09_zoom_19.tif',
            'aleppo_2016_08_03_zoom_19.tif',
            'aleppo_2016_09_18_zoom_19.tif',
            'aleppo_2016_10_19_zoom_19.tif',
            'aleppo_2016_11_03_zoom_19.tif',
            'aleppo_2017_02_20_zoom_19.tif',
            'aleppo_2017_08_03_zoom_19.tif',
	        'aleppo_2017_08_14_zoom_19.tif',
        ],
        'no_analysis': [
            '5_No_Analysis_Areas_Aleppo.shp',
        ],
    },
    'daraa': {
        'annotations': [
            '4_Damage_Sites_Daraa_CDA.shp',
        ],
        'rasters': [
            "daraa_2011_08_22_zoom_19.tif",
            "daraa_2011_10_17_zoom_19.tif",
            "daraa_2012_09_06_zoom_19.tif",
            "daraa_2013_11_10_zoom_19.tif",
            "daraa_2014_04_04_zoom_19.tif", #not complete  
	    "daraa_2014_05_01_zoom_19.tif",
	    "daraa_2014_06_03_zoom_19.tif",
            "daraa_2016_02_25_zoom_19.tif",
            "daraa_2016_04_19_zoom_19.tif",
	    "daraa_2016_07_18_zoom_19.tif",
            "daraa_2016_12_05_zoom_19.tif", 
            "daraa_2017_02_07_zoom_19.tif",
            "daraa_2017_06_02_zoom_19.tif", 
            "daraa_2017_10_03_zoom_19.tif",
        ],
        'no_analysis': [
            '5_No_Analysis_Areas_Daraa.shp',
        ],
    },
    'damascus': {
        'annotations': [
            	'Damage_Sites_Damascus_2017_Ex_Update.shp',
		#'4_Damage_Sites_Damascus_CDA.shp',
        ],
        'rasters': [
            #'damascus_2011_08_22_zoom_19.tif',
            'damascus_2012_10_10_zoom_19.tif',
            'damascus_2013_02_21_zoom_19.tif',
            'damascus_2015_09_26_zoom_19.tif',
            'damascus_2016_04_02_zoom_19.tif',
            #'damascus_2017_01_22_zoom_19.tif',
        ],
        'no_analysis': [
            '5_No_Analysis_Areas_Damascus.shp',
        ],
    },
    'raqqa': {
        'annotations': [
            '4_Damage_Sites_Raqqa_CDA.shp',
        ],
        'rasters': [
            'raqqa_2013_01_17_zoom_19.tif',
	    'raqqa_2013_02_13_zoom_19.tif', #bad
            'raqqa_2014_03_21_zoom_19.tif', #good
	    'raqqa_2014_05_18_zoom_19.tif', #good
	    'raqqa_2014_10_06_zoom_19.tif', #good
            'raqqa_2015_02_02_zoom_19.tif', #cloudy smoke?
            'raqqa_2016_07_01_zoom_19.tif', #good
	    'raqqa_2016_02_20_zoom_19.tif', #bad
	    'raqqa_2016_09_23_zoom_19.tif', #good
        ],
        'no_analysis': [
            '5_No_Analysis_Areas_Raqqa.shp',
        ],
    },
    'homs': {
        'annotations': [
            '4_Damage_Sites_Homs_CDA.shp',
        ],
        'rasters': [
            'homs_2016_05_30_zoom_19.tif',
            'homs_2014_04_03_zoom_19.tif', #good
            'homs_2013_10_31_zoom_19.tif', #good
            'homs_2011_05_21_zoom_19.tif',
	    'homs_2014_06_09_zoom_19.tif', #good 
	    'homs_2016_05_30_zoom_19.tif', #good
	    'homs_2016_07_05_zoom_19.tif',
	    #'homs_2017_06_18_zoom_19.tif',
	    #'homs_2018_03_19_zoom_19.tif',
        ],
        'no_analysis': [
            '5_No_Analysis_Areas_Homs.shp',
        ],
    },
    'hama': {
        'annotations': [
            '4_Damage_Sites_Hama_CDA.shp',
        ],
        'rasters': [
            'hama_2012_02_22_zoom_19.tif',
	    'hama_2012_07_17_zoom_19.tif', # new good
            'hama_2013_10_31_zoom_19.tif',
            'hama_2014_04_03_zoom_19.tif',
	        'hama_2014_05_25_zoom_19.tif',  # new bad
	        'hama_2014_06_13_zoom_19.tif', # new good
	        'hama_2016_05_18_zoom_19.tif',  # new bad
            'hama_2016_06_30_zoom_19.tif',
            'hama_2016_07_29_zoom_19.tif',
            'hama_2016_12_11_zoom_19.tif',  # new good
        ],
        'no_analysis': [
            '5_No_Analysis_Areas_Hama.shp',
        ],
    },
    'idlib': {
        'annotations': [
            '4_Damage_Sites_Idlib_CDA.shp',
        ],
        'rasters': [
            #"idlib_2011_07_31_zoom_19.tif",
            "idlib_2012_06_14_zoom_19.tif",
            "idlib_2014_02_07_zoom_19.tif",
            "idlib_2014_05_31_zoom_19.tif",
            "idlib_2015_08_19_zoom_19.tif",
            "idlib_2016_06_01_zoom_19.tif",
            "idlib_2016_08_01_zoom_19.tif",
            "idlib_2017_02_21_zoom_19.tif",
            "idlib_2017_07_17_zoom_19.tif",
        ],
        'no_analysis': [
            '5_No_Analysis_Areas_Idlib.shp',
        ],
    },
    'deir-ez-zor': {
        'annotations': [
            '4_Damage_Sites_Deir-ez-Zor_CDA.shp',
        ],
        'rasters': [
            "deir-ez-zor_2012_12_05_zoom_19.tif",
            "deir-ez-zor_2013_10_24_zoom_19.tif",
            "deir-ez-zor_2014_09_16_zoom_19.tif",
	    "deir-ez-zor_2014_11_25_zoom_19.tif",
	    "deir-ez-zor_2014_10_15_zoom_19.tif",
	    "deir-ez-zor_2016_04_17_zoom_19.tif",
            "deir-ez-zor_2016_05_25_zoom_19.tif",
	    "deir-ez-zor_2016_07_06_zoom_19.tif",
        ],
        'no_analysis': [
            '5_No_Analysis_Areas_Deir-ez-Zor.shp',
        ],
    },
}
