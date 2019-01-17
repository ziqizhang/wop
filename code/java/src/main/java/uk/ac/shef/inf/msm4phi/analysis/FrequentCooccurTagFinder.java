package uk.ac.shef.inf.msm4phi.analysis;

import com.opencsv.CSVWriter;
import no.uib.cipr.matrix.Matrix;
import no.uib.cipr.matrix.sparse.LinkedSparseMatrix;
import org.apache.commons.collections4.CollectionUtils;
import org.apache.commons.lang.exception.ExceptionUtils;
import org.apache.commons.lang3.tuple.ImmutablePair;
import org.apache.commons.lang3.tuple.Pair;
import org.apache.log4j.Logger;
import org.apache.solr.client.solrj.SolrClient;
import org.apache.solr.client.solrj.SolrQuery;
import org.apache.solr.client.solrj.response.QueryResponse;
import org.apache.solr.common.SolrDocument;
import uk.ac.shef.inf.msm4phi.Util;

import java.io.File;
import java.io.IOException;
import java.util.*;

public class FrequentCooccurTagFinder {
    private static final int resultBatchSize = 10000;
    private static final Logger LOG = Logger.getLogger(FrequentCooccurTagFinder.class.getName());

    private Set<String> exclusion = new HashSet<>();

    public FrequentCooccurTagFinder(String exclusionFile) throws IOException {
        if (exclusionFile != null)
            exclusion = Common.readExcludeList(exclusionFile, Integer.MAX_VALUE);
    }

    void process(SolrClient tweetCore, String hashtagFile, String outFile) throws IOException {
        Map<String, List<String>> disease2tagsInput = Util.readHashtags(new File(hashtagFile));
        calcCooccuringStats(disease2tagsInput, tweetCore, outFile);
    }

    private void calcCooccuringStats(Map<String, List<String>> disease2tagsInput,
                                     SolrClient tweetCore,
                                     String outFile) throws IOException {
        CSVWriter csvWriter = null;

        csvWriter = Util.createCSVWriter(outFile);

        for (Map.Entry<String, List<String>> entry : disease2tagsInput.entrySet()) {
            LOG.info("Processing disease=" + entry.getKey());
            calculate(csvWriter, tweetCore, 100, entry.getKey(), entry.getValue().toArray(new String[0]));
        }

        csvWriter.close();
    }


    private void calculate(CSVWriter csvWriter, SolrClient tweetCore, int topN, String disease,
                                                 String... diseaseTags) {
        Map<String, Long> counts=new HashMap<>();
        SolrQuery q = Util.createQueryTweetsOfHashtags(resultBatchSize, diseaseTags);
        boolean stop = false;
        long total = 0;
        while (!stop) {
            QueryResponse res = null;

            try {
                res = Util.performQuery(q, tweetCore);
                if (res != null)
                    total = res.getResults().getNumFound();
                //update results
                LOG.info(String.format("\t\ttotal results of %d, currently processing from %d to %d...",
                        total, q.getStart(), q.getStart() + q.getRows()));
                for (SolrDocument d : res.getResults()) {
                    Collection<Object> hashtags = d.getFieldValues("entities_hashtag");
                    if (hashtags != null) {
                        for (Object o : hashtags) {
                            String ht = o.toString();
                            if (ht.length() > 1) {
                                Long freq = counts.get(ht);
                                if (freq==null)
                                    freq=0L;
                                freq++;
                                counts.put(ht, freq);
                            }
                        }
                    }
                }

            } catch (Exception e) {
                LOG.warn(String.format("\t\tquery %s caused an exception: \n\t %s \n\t trying for the next query...",
                        q.toQueryString(), ExceptionUtils.getFullStackTrace(e)));
            }

            int curr = q.getStart() + q.getRows();
            if (curr < total)
                q.setStart(curr);
            else
                stop = true;

        }

        List<String> cooccuringTags = new ArrayList<>(counts.keySet());
        for(String t:diseaseTags)
            cooccuringTags.remove(Common.trimHashChar(t.toLowerCase()));
        Collections.sort(cooccuringTags, (s, t1) -> counts.get(t1).compareTo(counts.get(s)));
        int max = topN>cooccuringTags.size()?cooccuringTags.size():topN;
        for(int i=0; i<max; i++){
            String[] row = new String[4];
            row[0]=disease+"_"+i;
            row[1]=cooccuringTags.get(i);
            long freq = counts.get(row[1]);
            row[2]= String.valueOf(freq);
            row[3]= String.valueOf(freq/(double)total);
            csvWriter.writeNext(row);
        }
    }
}
