const API_URL = "/api"

export interface IntentResponse {
    intentType: string;
    message: string;
}

/**
 * Entry point for asking a question (without context)
 * 
 * @param query A question to ask (in natural English)
 * @returns A promise of an IntentResponse.
 */
export async function send(utterance: string): Promise<IntentResponse>
{
    const definitionURI = API_URL + "/chat";
    const payload = JSON.stringify({utterance});
    const headers = {'Content-Type': 'application/json'}
    const response = await fetch(definitionURI, {method: 'POST', headers: headers, body: payload});

    if (response.ok) {
        return await response.json() as IntentResponse;
    }
    else {
        console.warn(response.statusText);
        return Promise.resolve({intentType: "inconclusive", message: "No"} as IntentResponse);
    }
}

export async function createSession(): Promise<IntentResponse>
{
    const sessionURI = API_URL + "/chat";
    const response = await fetch(sessionURI);

    if (response.ok) {
        return await response.json() as IntentResponse;
    }
    else
    {
        console.warn(response.statusText);
        return Promise.reject();
        // return Promise.resolve({intentType: "inconclusive", response: "No"} as IntentResponse);
    }
}