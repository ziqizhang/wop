package uk.ac.shef.inf.msm4phi.analysis;

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

/**
 * co-occureence of two diseases = # unique users in both D1 and D2 / #unique users in either D1 and D2
 *
 */
public class GraphDiseaseCoocurByUser {
    private static final int resultBatchSize=10000;
    private static final Logger LOG = Logger.getLogger(GraphDiseaseCoocurByUser.class.getName());

    private Set<String> exclusion = new HashSet<>();

    public GraphDiseaseCoocurByUser(String exclusionFile) throws IOException {
        if (exclusionFile!=null)
            exclusion =Common.readExcludeList(exclusionFile,Integer.MAX_VALUE);
    }


    void process(SolrClient userCore, String hashtagFile, String outFile) throws IOException {
        Map<String, List<String>> disease2tagsInput= Util.readHashtags(new File(hashtagFile));
        Pair<Matrix, List<String>> output=calculateSimilarityMatrix(disease2tagsInput,userCore);
        Common.saveMatrixData(output.getLeft(), output.getRight(), outFile);
        Common.saveCommunityPairSimilarityData(output.getLeft(), output.getRight(), outFile+".pairs.csv");
    }

    private Pair<Matrix, List<String>> calculateSimilarityMatrix(Map<String, List<String>> disease2tagsInput,
                                                                 SolrClient userCore){
        Map<String, Set<String>> disease2AllUsers=new HashMap<>();
        Map<String, String> userID2nameMap = new HashMap<>();
        for(Map.Entry<String, List<String>> entry : disease2tagsInput.entrySet()){
            LOG.info("Processing disease="+entry.getKey());
            Set<String> userIDs= collectUsersOfDisease(userCore,userID2nameMap, entry.getValue().toArray(new String[0]));
            if (userIDs!=null &&userIDs.size()>0)
                disease2AllUsers.put(entry.getKey(), userIDs);
        }

        List<String> filteredDiseases = new ArrayList<>(disease2AllUsers.keySet());
        Collections.sort(filteredDiseases);
        Matrix m = new LinkedSparseMatrix(filteredDiseases.size(), filteredDiseases.size());

        double max=0.0;
        for(int i=0; i<filteredDiseases.size();i++){
            String diseaseA=filteredDiseases.get(i);
            Set<String> usersA =disease2AllUsers.get(diseaseA);
            System.out.println(i);
            for(int j=i+1; j<filteredDiseases.size();j++){
                String diseaseB=filteredDiseases.get(j);
                Set<String> usersB =disease2AllUsers.get(diseaseB);

                Collection<String> inter= CollectionUtils.intersection(usersA, usersB);
                double sim = (double)inter.size()/CollectionUtils.union(usersA, usersB).size();
                if (sim>max)
                    max=sim;
                m.set(i,j,sim);
                m.set(j,i, sim);
            }
        }

        //normalize(m, max);

        return new ImmutablePair<>(m, filteredDiseases);
    }

    private void normalize(Matrix m, double max) {
        for(int r=0;r<m.numRows();r++){
            for(int c=0;c<m.numColumns();c++)
                m.set(r,c, m.get(r,c)/max);
        }
    }

    private Set<String> collectUsersOfDisease(SolrClient userCore,Map<String,String> userID2NameMap,
                                                 String... diseaseTags){
        Set<String> userIDs =new HashSet<>();
        SolrQuery q=Util.createQueryUsersOfHashtags(resultBatchSize, diseaseTags);
        boolean stop = false;
        while (!stop) {
            QueryResponse res = null;
            long total = 0;
            try {
                res = Util.performQuery(q, userCore);
                if (res != null)
                    total = res.getResults().getNumFound();
                //update results
                LOG.info(String.format("\t\ttotal results of %d, currently processing from %d to %d...",
                        total, q.getStart(), q.getStart() + q.getRows()));
                for (SolrDocument d : res.getResults()) {
                    Object username = d.getFieldValue("user_screen_name");
                    if (username==null)
                        username=d.getFieldValue("id");

                    String userid = d.getFieldValue("id").toString();
                    userIDs.add(userid);
                    userID2NameMap.put(userid, username.toString());
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
        userIDs.removeAll(exclusion);
        return userIDs;
    }
}
