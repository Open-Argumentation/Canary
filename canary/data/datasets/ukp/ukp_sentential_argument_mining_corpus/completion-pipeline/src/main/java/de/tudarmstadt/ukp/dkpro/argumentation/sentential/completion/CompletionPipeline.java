package de.tudarmstadt.ukp.dkpro.argumentation.sentential.completion;

import java.io.File;
import java.nio.file.Files;
import java.util.Collection;
import java.util.HashMap;
import java.util.LinkedList;

import org.kohsuke.args4j.CmdLineException;
import org.kohsuke.args4j.CmdLineParser;
import org.kohsuke.args4j.Option;

import de.tudarmstadt.ukp.dkpro.argumentation.sentential.completion.preprocessing.Preprocessing;
import de.tudarmstadt.ukp.dkpro.argumentation.sentential.completion.retrieval.TextRetrieval;
import de.tudarmstadt.ukp.dkpro.argumentation.sentential.completion.tools.SentenceEntry;
import de.tudarmstadt.ukp.dkpro.argumentation.sentential.completion.tools.TSVTools;

/**
 * This class downloads the annotated sentences from the web and stores them in
 * a TSV-file
 * 
 * @author Christian Stab and Debanjan Chaudhuri
 */
public class CompletionPipeline {

	@Option(name = "-i", aliases = {
			"--inputPath" }, metaVar = "string", usage = "path to incomplete TSV files", required = true)
	public static String input;

	@Option(name = "-o", aliases = {
			"--outputPath" }, metaVar = "string", usage = "path where the completes TSV files are stored", required = true)
	public static String output;

	private static int retries = 5;

	private static String lang = "en";

	public static void main(String[] args) throws Exception {
		new CompletionPipeline().doMain(args);
	}

	private void doMain(String[] args) throws Exception {
		CmdLineParser parser = new CmdLineParser(this);
		try {
			parser.parseArgument(args);
		} catch (CmdLineException e) {
			System.err.println(e.getMessage());
			System.err.println(this.getClass().getSimpleName() + " [options]");
			parser.printUsage(System.err);
			System.err.println();
			return;
		}

		long start = System.currentTimeMillis();

		// read all the files kept in the tsv input path
		File inputPath = new File(input);
		File outputPath = new File(output);
		if (!outputPath.exists() || !outputPath.isDirectory()) {
			Files.createDirectories(outputPath.toPath());
		}
		for (File f : inputPath.listFiles()) {
			if (!f.getName().endsWith(".tsv"))
				continue;
			System.out.println("Processing file '" + f.getName() + "'");
			Collection<SentenceEntry> entries = TSVTools.readTSV(f);
			CompletionPipeline pipeline = new CompletionPipeline();
			boolean status = pipeline.completeEntries(entries);
			if (status) {
				TSVTools.writeTSV(new File(outputPath.getAbsolutePath() + "/" + f.getName()), entries, true);
			}
		}

		long end = System.currentTimeMillis();
		System.out.println("Total run time: " + (end - start) + " ms");
	}

	public boolean completeEntries(Collection<SentenceEntry> entries) {
		boolean status = true;

		HashMap<String, LinkedList<SentenceEntry>> map = getUrlBuckets(entries);

		for (String url : map.keySet()) {
			LinkedList<SentenceEntry> list = map.get(url);
			System.out.println("  Retrieving text from '" + url + "'");
			while (!completeEntries(url, list)) {
			}
			;
		}

		return status;
	}

	/**
	 * Retrieves all sentences from the given URL and adds them to the given
	 * SentenceEntries
	 * 
	 * @param url,
	 *            not null
	 * @param entries,
	 *            not null
	 * @return true, if successful completion
	 */
	private boolean completeEntries(String url, LinkedList<SentenceEntry> entries) {
		String originalURL = entries.get(0).retrievedUrl;

		// retrieve content from url
		String content = null;

		try {
			content = TextRetrieval.getContentFromURL(url, originalURL, retries);
		} catch (Exception e) {
			e.printStackTrace();
			System.out.println("  Warning: Couldn't retrieve text");
			System.out.println("    URL: " + url);
			System.out.println("    Original URL: " + originalURL);
			return false;
		}

		// complete entries
		HashMap<String, String> retrievedSentences = Preprocessing.extractSentences(content, lang);
		for (SentenceEntry entry : entries) {
			if (!retrievedSentences.containsKey(entry.sentenceHash)) {
				System.out.println("  Warning: Couldn't find sentence '" + entry.sentenceHash + "'  ");
				return false;
			} else {
				entry.sentence = retrievedSentences.get(entry.sentenceHash);
			}
		}

		return true;
	}

	/**
	 * Returns a HashMap with the webArchive URL as key and for each URL a list of
	 * all SentenceEntries that belong to that URL.
	 * 
	 * @param entries
	 * @return HashMap
	 */
	private static HashMap<String, LinkedList<SentenceEntry>> getUrlBuckets(Collection<SentenceEntry> entries) {
		HashMap<String, LinkedList<SentenceEntry>> result = new HashMap<String, LinkedList<SentenceEntry>>();
		for (SentenceEntry s : entries) {
			if (!result.containsKey(s.archiveUrl)) {
				result.put(s.archiveUrl, new LinkedList<SentenceEntry>());
			}
			result.get(s.archiveUrl).add(s);
		}

		return result;
	}

}
