package uk.ac.shef.inf.msm4phi.stats.user;

import com.opencsv.CSVReader;

import java.io.IOException;
import java.io.Reader;
import java.io.Writer;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.concurrent.TimeUnit;

import com.opencsv.CSVWriter;
import org.apache.log4j.Logger;

import org.apache.commons.lang.exception.ExceptionUtils;
import twitter4j.*;
import uk.ac.shef.inf.msm4phi.Util;

/**
 * given a time window, and a list of users, count for each one how many tweets are created within that month
 * <p>
 * (uses api)
 * <p>
 * start: 20-04 18:10
 * end: 20-05 18:10
 */


public class UserTotalTweetsCounter {

    public static void main(String[] args) throws ParseException, IOException {
        Date start=new SimpleDateFormat("dd/MM/yyyy HH:mm").parse("20/04/2018 18:10");
        Date end=new SimpleDateFormat("dd/MM/yyyy HH:mm").parse("20/05/2018 18:10");

        UserTotalTweetsCounter utc = new UserTotalTweetsCounter(start,end,
                args[0],args[1],args[2],args[3]);
        List<String[]> users=utc.readUsers(args[4],0,1,26,27);
        utc.countTweets(users, args[5]);
    }



    private static final Logger LOG = Logger.getLogger(UserTotalTweetsCounter.class.getName());
    private Twitter twitter;
    private Date start;
    private Date end;
    public UserTotalTweetsCounter(Date start, Date end,
            String twitterConsumerKey, String twitterConsumerSecret,
            String twitterAccessToken, String twitterAccessSecret){
        this.twitter = Util.authenticateTwitter(twitterConsumerKey, twitterConsumerSecret, twitterAccessToken,
                twitterAccessSecret);
        this.start=start;
        this.end=end;
    }

    public void countTweets(List<String[]> users, String outFile) throws IOException {
        Writer writer = Files.newBufferedWriter(Paths.get(outFile));

        CSVWriter csvWriter = new CSVWriter(writer,
                CSVWriter.DEFAULT_SEPARATOR,
                CSVWriter.NO_QUOTE_CHARACTER,
                CSVWriter.DEFAULT_ESCAPE_CHARACTER,
                CSVWriter.DEFAULT_LINE_END);

        String[] header=new String[5];
        header[0]="user_id";
        header[1]="screenname";
        header[2]="new tweets_with_hashtags";
        header[3]="retweets_with_hashtags";
        header[4]="feature_ratio";
        csvWriter.writeNext(header);

        int countU=0;
        for(String[] u : users){
            countU++;

            String screenname=u[1];

            LOG.info(String.format("processing user %s, %s",
                    screenname, String.valueOf(countU)));

            int page=1, tweets_in_range=0, tweets_total=0;
            Date total_end=null, total_start=null;

            while(true) {
                LOG.info(String.format("\t\tpage %s",
                        page));
                Paging paging = new Paging(page, 200);
                ResponseList<Status> statuses = null;
                try {
                    statuses = twitter.getUserTimeline(screenname, paging);
                    Thread.sleep(1000);
                } catch (Exception e) {
                    LOG.warn(String.format("\t\tfailed to count user tweets %s \n\t %s, trying again in 10 secs...",
                            screenname, ExceptionUtils.getFullStackTrace(e)));
                    break;
                }

                if (total_end==null &&statuses.size()>0)
                    total_end=statuses.get(0).getCreatedAt();
                tweets_total+=statuses.size();
                boolean stop=false;
                if(statuses!=null) {
                    for (Status s : statuses) {
                        if (start.after(s.getCreatedAt())){
                            stop=true;
                            break;
                        }
                        if (s.getCreatedAt().after(end))
                            continue;

                        tweets_in_range++;
                    }
                }

                page++;
                if (statuses.size()>0)
                    total_start=statuses.get(statuses.size()-1).getCreatedAt();
                if (stop)
                    break;
                if (statuses.size()==0 && total_start.after(start)){
                    long diff = total_end.getTime() - total_start.getTime();
                    long days=diff / (24 * 60 * 60 * 1000) + 1;
                    long tweets_per_day=tweets_total/days;
                    tweets_in_range=(int)tweets_per_day*30;
                    LOG.info("\t\tstats estimated for this user %s");
                    break;
                }
            }

            int total_stored_tweets=Integer.valueOf(u[2])+Integer.valueOf(u[3]);
            if(tweets_in_range==0){
                System.err.println(String.format("Cannot find tweets within the time window for %s", screenname));
                tweets_in_range=total_stored_tweets;
            }else if (tweets_in_range<total_stored_tweets){
                System.err.println(String.format("Found tweets less than stored number within the time window for %s", screenname));
                tweets_in_range=total_stored_tweets;
            }
            u[4]=String.valueOf((double) total_stored_tweets/tweets_in_range);

            csvWriter.writeNext(u);
            csvWriter.flush();
        }


        //save result
        csvWriter.close();
    }

    static List<String[]> readUsers(String inputCSV, int colID, int colScreename,
                                     int colNT, int colRT) throws IOException {
        Reader reader = Files.newBufferedReader(Paths.get(inputCSV));
        CSVReader csvReader = new CSVReader(reader);

        // Reading Records One by One in a String array
        List<String[]> res = new ArrayList<>();
        String[] nextRecord;
        int count=0;
        while ((nextRecord = csvReader.readNext()) != null) {
            if (count==0) {
                count++;
                continue;
            }
            String[] row = new String[5];
            row[0]=nextRecord[colID];
            row[1]=row[0].substring(row[0].lastIndexOf("/")+1);
            row[2]=nextRecord[colNT];
            row[3]=nextRecord[colRT];
            row[4]="0";
            res.add(row);
        }
        return res;
    }


}
