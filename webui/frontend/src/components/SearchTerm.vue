<template>
    <div>
        <b-form @submit="onSubmit">
            <b-form-input
                id="term-input"
                v-model="keyword"
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
    </div>
</template>

<script lang="ts">

import Vue from 'vue';

export default Vue.extend({
    name: "SearchTerm",

    data() {
        return {
            keyword: "",
            searching: false
        }
    },

    methods: {
        onSubmit(event: Event): void {
            event.preventDefault();
            console.log(this.keyword);

            this.searching = true;
            setTimeout( () => {this.searching = false}, 5000);
        }
    }
});
</script>
