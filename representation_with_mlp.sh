varios_datasets(){    		
    for dataset in $1; do        		
        for index in $2; do            
            python representation_with_mlp.py ${dataset} ${index} $3
        done
    done
}

#varios_datasets 'sst2' '0 1 2 3 4 5 6 7 8 9'
#varios_datasets 'sogou yelp_2015' '0 1 2 3 4'
#varios_datasets 'imdb_reviews sogou yelp_2015' '3 4'
varios_datasets 'aisopos_ntua_2L' '0' '10'