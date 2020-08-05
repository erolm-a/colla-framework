const API_URL = window.location.hostname.startsWith("127.0.0.1") ? "http://127.0.0.1:8080/api" : "/api"
console.log(window.location.hostname)

export interface IntentResponse {
    intentType: string;
    message: string;
}

export type EntityResponse = string;
export type Entity = any;

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

export async function searchByLabel(label: string): Promise<Entity[]>
{
    const fetchURI = API_URL + `/search/${label}`;
    const response = await fetch(fetchURI);

    if (response.ok) {
        return await response.json() as EntityResponse[];
    }
    else
    {
        console.warn(response.statusText);
        return Promise.reject();
    }
}

export async function searchKGItem(item: string): Promise<EntityResponse>
{
    console.assert(item.startsWith("kgl:") ||  item.startsWith("kglprop"));


    const fetchURI = API_URL + `/kg/${item}`;
    const response = await fetch(fetchURI);

    if (response.ok) {
        return await response.json() as EntityResponse
    }
    else
    {
        console.warn(response.statusText);
        return Promise.reject();
    }
}

export function stripPrefix(value: string): string
{
    const kglNamespace = "http://grill-lab.org/kg/entity/";
    const kglpropNamespace = "http://grill-lab.org/kg/property/"


    if(value.startsWith(kglNamespace))
    {
      return "kgl:" + value.slice(kglNamespace.length);
    }
    else if(value.startsWith(kglpropNamespace))
    {
      return "kglprop:" + value.slice(kglpropNamespace.length);
    }

    return value;

}