package uk.ac.shef.inf.msm4phi.analysis;

import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.lang3.tuple.Pair;
import org.apache.commons.math3.stat.StatUtils;
import org.apache.log4j.Logger;
import org.apache.solr.client.solrj.SolrClient;

import java.io.IOException;
import java.io.PrintWriter;
import java.util.*;

/**
 * calculate:
 *
 * % of users for each hastag
 * print top 100 above
 * print min, max, 75%, 50%, 25%, mean
 */
public class HashtagUserAwareness {
    private static final int resultBatchSize = 10000;
    private static final Logger LOG = Logger.getLogger(HashtagUserAwareness.class.getName());

    void process(String outFile,
                 SolrClient userCore) throws IOException {
        LOG.info("Calculating users in multiple disease communities...");
        Pair<Map<String, Long>, Long> resPair
                = Common.findHashtagsAndUsers(userCore, LOG, resultBatchSize);

        Map<String, Long> res = resPair.getKey();
        //hashtags found in at least 2 users
        Map<String, Long> selected = new HashMap<>();
        List<Double> userNum = new ArrayList<>();
        for(Map.Entry<String, Long> en:res.entrySet()){
            if(en.getValue()>1){
                selected.put(en.getKey(), en.getValue());
                userNum.add((double)en.getValue());
            }
        }

        Collections.sort(userNum);
        Collections.reverse(userNum);
        List<String> tags = new ArrayList<>(selected.keySet());
        tags.sort((s, t1) ->
                Long.compare(selected.get(t1), selected.get(s)));
        double[] values = ArrayUtils.toPrimitive(userNum.toArray(new Double[0]));


        PrintWriter p = new PrintWriter(outFile);
        StringBuilder sb = new StringBuilder();
        sb.append("total unique tags,").append(res.size()).append("\n")
                .append("multi-user awareness as %,").append((double)selected.size()/res.size()).append("\n")
                .append("max tag awareness,").append(StatUtils.max(values)).append("\n")
                .append(".75 quantile,").append(StatUtils.percentile(values, 0.75)).append("\n")
                .append(".50 quantile (median),").append(StatUtils.percentile(values, 0.50)).append("\n")
                .append(".25 quantile,").append(StatUtils.percentile(values, 0.25)).append("\n")
                .append("min tag awareness,").append(StatUtils.min(values)).append("\n")
                .append("average,").append(StatUtils.mean(values)).append("\n\n")
                .append("Top 500 by tag awareness (as #of users)\n");

        for (int i=0; i<500; i++){
            sb.append(tags.get(i)).append(",").append(userNum.get(i)/(double)resPair.getRight()).append("\n");
        }
        p.println(sb.toString());
        p.close();


    }
}
