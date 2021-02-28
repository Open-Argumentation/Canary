package de.tudarmstadt.ukp.dkpro.argumentation.sentential.completion.retrieval;

/**
 * Class for returning the content and the response code from the html page
 * processing
 * 
 * @author Christian Stab and Debanjan Chaudhuri
 *
 */
public class HttpResponse {
	public String htmlContent;
	public int statusCode;
	public String htmlOrigBody;
}
