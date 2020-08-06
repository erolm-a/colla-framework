<template>
    <div style="text-align: left">
        <b-form @submit="onSubmit" inline class="d-flex mb-2">
            <b-form-input
                id="term-input"
                v-model="keyword"
                class="mb-2 mr-sm-2 mb-sm-0 flex-fill"
                required
                placeholder="Enter the term to search">
            </b-form-input>
            <b-button variant="primary" disabled v-if="searching">
                <b-spinner small type="grow" />
                Loading...
            </b-button>
            <b-button variant="primary" type="submit" v-else>
                Search
            </b-button>
        </b-form>

        <b-alert :show=showErrorAlert variant="danger">
            {{error}}
        </b-alert>
        <ul class="list-group">
            <li class="list-group-item" v-for="result in results" :key="result">
                <router-link :to="'/kg/'+result">{{result}}</router-link>
            </li>
        </ul>
    </div>
</template>

<script lang="ts">
import Vue from 'vue';
import {searchByLabel, EntityResponse} from '../api';
export default Vue.extend({
    name: "KGSearch",
    data() {
        return {
            keyword: "",
            results: [] as EntityResponse[],
            searching: false,
            error: "",
            showErrorAlert: false
        }
    },
    methods: {
        async onSubmit(event: Event) {
            event.preventDefault();
            this.searching = true;
            try {
                this.error = ""
                this.showErrorAlert = false;
                const response = await searchByLabel(this.keyword);
                console.log(response);
                this.results = response;
                if(this.results.length == 0)
                {
                    throw Error("The search did not return any result");
                }
            }
            catch (error)
            {
                this.showErrorAlert = true;
                this.error = error;
            }
            this.searching = false;
        }
    }
});
</script>